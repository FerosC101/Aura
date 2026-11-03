"""The AURA Engine: the real-time pipeline.

    microphone -> buffering -> preprocessing -> separation -> classification
    -> speaker identification -> relevance scoring -> gain assignment
    -> reconstruction -> headphones

Threading model
---------------
Two threads, split by deadline rather than by module:

**The audio callback** runs separation, classification, policy and mixing. All
of it is bounded and fast (~1 ms against a 20 ms budget), and doing it inline
avoids the extra block of buffering a hand-off to a worker would cost.

**The semantic worker** runs Person Lock embedding and Name Awareness. Those
need a 1-1.5 s window and take several milliseconds -- a spike that would blow
the audio deadline and produce an audible dropout. They are also inherently
slow-rate decisions: who is speaking does not change every 20 ms. The worker
publishes its conclusions into plain attributes that the audio thread reads;
assignment of an immutable result object is atomic under the GIL, so no lock is
needed on the real-time path, and a stale-by-one-window answer is harmless.

Everything the UI needs is published as an :class:`~aura.types.AcousticScene`
at a fixed rate, decoupled from the audio rate so a slow subscriber can never
stall audio.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable

import numpy as np

from ..audio.buffer import AudioBufferManager
from ..audio.capture import (
    BaseAudioEngine,
    DuplexAudioEngine,
    OfflineAudioEngine,
    SplitAudioEngine,
)
from ..audio.mixer import AudioMixer, MixReport
from ..audio.preprocess import AudioPreprocessor
from ..classification.registry import (
    build_classifier,
    build_separator,
    build_speaker_identifier,
    fallback_notes,
)
from ..classification.taxonomy import get_category
from ..config import EngineConfig
from ..context.engine import ContextEngine, MotionHint
from ..metrics.monitor import MetricsMonitor
from ..policy.engine import AcousticPolicyEngine, PolicyContext
from ..policy.modes import get_mode
from ..policy.safety import SafetyGuard
from ..profile.acoustic_profile import AcousticProfile, ProfileStore
from ..profile.learning import PreferenceLearningEngine
from ..speaker.person_lock import PersonLockManager, SpeakerProfile
from ..speech.name_awareness import NameAwareness, TemplateKeywordSpotter
from ..types import (
    AcousticScene,
    AudioFrame,
    ContextState,
    GainDecision,
    ListeningMode,
    SeparatedAudio,
    SoundEvent,
    StreamKind,
)

SceneSubscriber = Callable[[AcousticScene], None]

#: Speech-stream level above which the speaker/name workers consider a frame
#: to contain speech worth analysing.
SPEECH_GATE_DBFS = -52.0


class AuraEngine:
    """Owns every component and runs the real-time loop."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        profile: AcousticProfile | None = None,
        learner: PreferenceLearningEngine | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        sr, hop = self.config.sample_rate, self.config.block_size

        # -- persisted preferences ----------------------------------------
        self.store = ProfileStore(self.config.profile_dir)
        if profile is None or learner is None:
            loaded_profile, loaded_learner = self.store.load()
            self.profile = profile or loaded_profile
            self.learner = learner or loaded_learner
        else:
            self.profile, self.learner = profile, learner

        # -- DSP / AI components ------------------------------------------
        self.preprocessor = AudioPreprocessor(sr)
        self.separator = build_separator(self.config.separator_backend, sr, hop)
        self.classifier = build_classifier(
            self.config.classifier_backend, sr,
            threshold=self.config.classifier_threshold,
            model_path=self.config.classifier_model_path,
        )
        self.identifier = build_speaker_identifier(self.config.speaker_backend, sr)

        safety_set = (
            set(self.config.safety_categories)
            if self.config.safety_categories is not None
            else (set(self.profile.safety_categories) or None)
        )
        self.safety = SafetyGuard(enabled_categories=safety_set)
        self.policy = AcousticPolicyEngine(self.safety)
        self.mixer = AudioMixer(sr, output_trim_db=self.config.output_trim_db)
        self.buffers = AudioBufferManager(sr)
        self.metrics = MetricsMonitor(sr, hop)
        self.context = ContextEngine(
            auto_enabled=self.config.auto_context or self.profile.privacy.auto_context_detection,
            allow_motion=self.config.allow_motion_context
            or self.profile.privacy.allow_motion_context,
        )

        self.person_lock = PersonLockManager(self.identifier, sr)
        self.name_awareness = NameAwareness(
            TemplateKeywordSpotter(sr), sr,
            enabled=self.config.name_awareness_enabled
            and self.profile.privacy.allow_name_awareness,
        )
        self._restore_speaker_profiles()

        # -- runtime state -------------------------------------------------
        self.mode: ListeningMode = self.config.mode
        self.enabled = self.config.enabled
        self.suppression_strength = self.config.suppression_strength

        self._audio: BaseAudioEngine | None = None
        self._sequence = 0
        self._scene: AcousticScene | None = None
        self._last_mix: MixReport = MixReport()
        self._subscribers: list[SceneSubscriber] = []
        self._sub_lock = threading.Lock()
        self._last_publish = 0.0
        self._mode_since = time.time()
        self._running = False

        # Published by the semantic worker, read by the audio thread.
        self._lock_result = self.person_lock.last
        self._name_active = False

        # Audio thread -> worker hand-off.
        self._worker: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._speech_pending: list[np.ndarray] = []
        self._speech_present = False
        self._speech_lock = threading.Lock()

        self.metrics.set_algorithmic_delay(self.separator.algorithmic_delay_samples)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, *, signal: np.ndarray | None = None, realtime: bool = True) -> None:
        """Open the audio device, or drive the pipeline from ``signal``."""
        if self._running:
            return
        cfg = self.config

        if signal is not None:
            self._audio = OfflineAudioEngine(
                cfg.sample_rate, cfg.block_size, self._process, signal,
                realtime=realtime, loop=True,
            )
        elif cfg.duplex:
            self._audio = DuplexAudioEngine(
                cfg.sample_rate, cfg.block_size, self._process,
                input_device=cfg.input_device, output_device=cfg.output_device,
                latency=cfg.device_latency,
            )
        else:
            self._audio = SplitAudioEngine(
                cfg.sample_rate, cfg.block_size, self._process,
                input_device=cfg.input_device, output_device=cfg.output_device,
            )

        self._running = True
        self._worker_stop.clear()
        self._worker = threading.Thread(
            target=self._semantic_loop, name="aura-semantic", daemon=True
        )
        self._worker.start()

        self._audio.start()
        self.metrics.set_device_latency(
            self._audio.input_latency_ms, self._audio.output_latency_ms
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._worker_stop.set()
        if self._audio is not None:
            self._audio.stop()
            self._audio = None
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None
        self._note_mode_usage()

    def close(self) -> None:
        self.stop()
        self.classifier.close()
        self.separator.close()
        self.identifier.close()
        self.save_profile()

    @property
    def running(self) -> bool:
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # The real-time path
    # ------------------------------------------------------------------

    def _process(self, in_block: np.ndarray, out_block: np.ndarray) -> None:
        """Called once per block from the audio thread. Must not block."""
        wall_start = time.perf_counter()
        self.metrics.begin_frame()
        self._sequence += 1

        with self.metrics.stage("preprocess"):
            samples = self.preprocessor.process(in_block)
        frame = AudioFrame(samples, self.config.sample_rate, sequence=self._sequence)

        with self.metrics.stage("separation"):
            separated = self.separator.separate(samples)

        with self.metrics.stage("classification"):
            events = self.classifier.classify(frame, separated)

        # Speaker identification and name awareness are *observed* here and
        # *decided* on the worker thread; this is only the hand-off.
        with self.metrics.stage("speaker"):
            self._handoff_speech(separated)
            lock = self._lock_result
            name_active = self._name_active

        with self.metrics.stage("policy"):
            ctx = self._policy_context(frame, lock.active, lock.name, name_active)
            decisions = self.policy.decide(events, ctx)

        with self.metrics.stage("mixing"):
            mixed, report = self.mixer.mix(separated, decisions, bypass=not self.enabled)

        n = min(out_block.size, mixed.size)
        out_block[:n] = mixed[:n]
        if n < out_block.size:
            out_block[n:] = 0.0

        self._last_mix = report
        self.buffers.push(samples, out_block)
        self.context.observe(events, frame.dbfs)

        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        self.metrics.end_frame(wall_ms=wall_ms)
        if self._audio is not None and self._audio.xruns > self.metrics.xruns:
            self.metrics.record_xrun(self._audio.xruns - self.metrics.xruns)

        self._publish(events, decisions, frame, report, lock, name_active)

    def _policy_context(
        self,
        frame: AudioFrame,
        lock_active: bool,
        lock_name: str | None,
        name_active: bool,
    ) -> PolicyContext:
        return PolicyContext(
            mode=self.mode,
            context=self.context._last.state,
            person_lock_active=lock_active and self.config.person_lock_enabled,
            person_lock_name=lock_name,
            name_detected=name_active,
            user_overrides=self.profile.overrides_for(self.mode),
            learned_bias=self.learner.biases_for_mode(self.mode),
            suppression_strength=self.suppression_strength,
            enabled=self.enabled,
            scene_level_db=frame.dbfs,
        )

    def _handoff_speech(self, separated: SeparatedAudio) -> None:
        """Queue the speech stream for the worker. Never blocks meaningfully."""
        speech = separated.streams.get(StreamKind.PRIMARY_SPEECH)
        if speech is None or speech.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(speech, dtype=np.float64)) + 1e-12))
        present = 20.0 * np.log10(max(rms, 1e-9)) > SPEECH_GATE_DBFS
        with self._speech_lock:
            self._speech_present = present
            if present:
                self._speech_pending.append(speech.copy())
                # Bound the hand-off: if the worker is behind, drop the oldest
                # audio rather than let the queue grow without limit.
                if len(self._speech_pending) > 200:
                    del self._speech_pending[:-100]

    # ------------------------------------------------------------------
    # The semantic worker
    # ------------------------------------------------------------------

    def _semantic_loop(self) -> None:
        """Person Lock and Name Awareness, off the real-time path."""
        while not self._worker_stop.is_set():
            with self._speech_lock:
                pending = self._speech_pending
                self._speech_pending = []
                present = self._speech_present

            if not pending:
                self._worker_stop.wait(0.02)
                continue

            try:
                for block in pending:
                    if self.config.person_lock_enabled:
                        self._lock_result = self.person_lock.update(block, present)
                    if self.name_awareness.enabled:
                        self._name_active = self.name_awareness.update(block, present)
            except Exception:
                # A failure in an optional feature must never take audio down.
                self._lock_result = self.person_lock.last
            if not self.name_awareness.enabled:
                self._name_active = False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def subscribe(self, fn: SceneSubscriber) -> Callable[[], None]:
        with self._sub_lock:
            self._subscribers.append(fn)

        def unsubscribe() -> None:
            with self._sub_lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return unsubscribe

    def _publish(
        self,
        events: list[SoundEvent],
        decisions: list[GainDecision],
        frame: AudioFrame,
        report: MixReport,
        lock,
        name_active: bool,
    ) -> None:
        now = time.perf_counter()
        interval = 1.0 / max(self.config.publish_rate_hz, 1e-3)
        scene = AcousticScene(
            events=events[: self.config.max_reported_events],
            decisions=decisions[: self.config.max_reported_events],
            mode=self.mode,
            context=self.context._last.state,
            person_lock_active=lock.active and self.config.person_lock_enabled,
            person_lock_name=lock.name,
            person_lock_similarity=lock.similarity,
            name_detected=name_active,
            safety_override_active=self.policy.last_safety_override,
            input_level_db=report.input_level_db,
            output_level_db=report.output_level_db,
            sequence=frame.sequence,
        )
        self._scene = scene
        if now - self._last_publish < interval:
            return
        self._last_publish = now

        with self._sub_lock:
            subscribers = list(self._subscribers)
        for fn in subscribers:
            # A broken subscriber must not disturb the audio thread.
            with contextlib.suppress(Exception):
                fn(scene)

    # ------------------------------------------------------------------
    # Control surface
    # ------------------------------------------------------------------

    def set_mode(self, mode: ListeningMode | str) -> ListeningMode:
        new = ListeningMode(mode) if isinstance(mode, str) else mode
        if new is not self.mode:
            self._note_mode_usage()
            self.mode = new
            self.classifier.reset()
        return self.mode

    def _note_mode_usage(self) -> None:
        now = time.time()
        self.profile.note_mode_use(self.mode, now - self._mode_since)
        self._mode_since = now

    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.mixer.reset()
        return self.enabled

    def set_suppression_strength(self, value: float) -> float:
        self.suppression_strength = float(np.clip(value, 0.0, 1.5))
        self.profile.suppression_strength = self.suppression_strength
        return self.suppression_strength

    def set_category_gain(
        self, category: str, gain_db: float | None, *, learn: bool = True
    ) -> dict:
        """Move one mixer slider, and learn from the correction.

        The learning signal is the difference between what the user chose and
        what the policy would have chosen unaided -- the correction itself --
        so the lesson stays valid even if the mode's baseline changes later.
        """
        if learn and gain_db is not None:
            baseline = self.policy.explain(category, self._policy_context(
                AudioFrame(np.zeros(1, dtype=np.float32), self.config.sample_rate),
                self._lock_result.active, self._lock_result.name, self._name_active,
            ))
            self.learner.observe_adjustment(
                self.mode, category, float(gain_db) - float(baseline["gain_db"])
            )
        self.profile.set_override(self.mode, category, gain_db)
        return self.mixer_state()

    def mixer_state(self) -> dict:
        """Current slider positions and what each category would be doing."""
        ctx = self._policy_context(
            AudioFrame(np.zeros(1, dtype=np.float32), self.config.sample_rate),
            self._lock_result.active, self._lock_result.name, self._name_active,
        )
        overrides = self.profile.overrides_for(self.mode)
        rows = []
        for key in self.classifier.categories:
            meta = get_category(key)
            explained = self.policy.explain(key, ctx)
            rows.append({
                **explained,
                "override_db": overrides.get(key),
                "safety_critical": meta.safety_critical,
                "stream": meta.stream.value,
                "icon": meta.icon,
                "description": meta.description,
                "learned_db": round(self.learner.bias_for(self.mode, key), 2),
            })
        return {"mode": self.mode.value, "categories": rows}

    # -- Person Lock ---------------------------------------------------

    def enrol_speaker(
        self, name: str, clips: list[np.ndarray], sample_rate: int, *, consent: bool = True
    ) -> SpeakerProfile:
        profile = self.person_lock.enrol(name, clips, sample_rate, consent=consent)
        if self.profile.privacy.store_speaker_embeddings:
            self.profile.speaker_profiles = [
                p.to_dict() for p in self.person_lock.profiles.values()
            ]
            self.profile.selected_speaker_id = self.person_lock.selected_id
            self.save_profile()
        return profile

    def select_speaker(self, profile_id: str | None) -> bool:
        ok = self.person_lock.select(profile_id)
        if ok:
            self.profile.selected_speaker_id = profile_id
            self._lock_result = self.person_lock.last
        return ok

    def delete_speaker(self, profile_id: str) -> bool:
        ok = self.person_lock.delete(profile_id)
        if ok:
            self.profile.speaker_profiles = [
                p.to_dict() for p in self.person_lock.profiles.values()
            ]
            self.profile.selected_speaker_id = self.person_lock.selected_id
            self.save_profile()
        return ok

    def _restore_speaker_profiles(self) -> None:
        for raw in self.profile.speaker_profiles:
            try:
                p = SpeakerProfile.from_dict(raw)
            except Exception:
                continue
            # A stored embedding from another backend is not comparable.
            if p.backend and p.backend != self.identifier.name:
                continue
            self.person_lock.profiles[p.id] = p
        if self.profile.selected_speaker_id in self.person_lock.profiles:
            self.person_lock.selected_id = self.profile.selected_speaker_id
        elif self.person_lock.profiles:
            self.person_lock.selected_id = next(iter(self.person_lock.profiles))

    # -- Name Awareness ------------------------------------------------

    def enrol_name(self, word: str, clips: list[np.ndarray], sample_rate: int) -> bool:
        ok = self.name_awareness.spotter.enrol(word, clips, sample_rate)
        if ok and word not in self.profile.trigger_words:
            self.profile.trigger_words.append(word)
            self.save_profile()
        return ok

    def set_name_awareness(self, enabled: bool) -> bool:
        self.name_awareness.enabled = bool(enabled) and self.profile.privacy.allow_name_awareness
        if not self.name_awareness.enabled:
            self._name_active = False
            self.name_awareness.reset()
        return self.name_awareness.enabled

    # -- Context -------------------------------------------------------

    def set_context(self, state: ContextState | str | None) -> dict:
        self.context.declare(state)
        return self.context.snapshot()

    def submit_motion(self, hint: MotionHint) -> bool:
        return self.context.submit_motion(hint)

    # -- Persistence ---------------------------------------------------

    def save_profile(self) -> str:
        return str(self.store.save(self.profile, self.learner))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def scene(self) -> AcousticScene | None:
        return self._scene

    def scene_dict(self) -> dict:
        scene = self._scene
        if scene is None:
            return {"events": [], "decisions": [], "mode": self.mode.value}
        by_category = {d.category: d for d in scene.decisions}
        cards = []
        for e in scene.events:
            d = by_category.get(e.category)
            meta = get_category(e.category)
            cards.append({
                "category": e.category,
                "label": meta.label,
                "icon": meta.icon,
                "confidence": round(e.confidence, 3),
                "level_db": round(e.level_db, 1),
                "stream": e.stream.value,
                "safety_critical": meta.safety_critical,
                "action": d.action.value if d else "preserve",
                "gain_db": round(d.gain_db, 1) if d else 0.0,
                "priority": d.priority.value if d else "normal",
                "safety_override": bool(d and d.safety_override),
                "reason": d.reason if d else "",
            })
        return {
            "sequence": scene.sequence,
            "timestamp": scene.timestamp,
            "mode": scene.mode.value,
            "context": scene.context.value,
            "enabled": self.enabled,
            "sounds": cards,
            "person_lock": {
                "active": scene.person_lock_active,
                "name": scene.person_lock_name,
                "similarity": round(scene.person_lock_similarity, 3),
            },
            "name_awareness": {
                "enabled": self.name_awareness.enabled,
                "active": scene.name_detected,
                "words": self.name_awareness.spotter.words,
            },
            "safety_override_active": scene.safety_override_active,
            "levels": {
                "input_db": round(scene.input_level_db, 1),
                "output_db": round(scene.output_level_db, 1),
            },
            "mix": self._last_mix.as_dict(),
        }

    def status(self) -> dict:
        audio = self._audio.describe() if self._audio else {"transport": None, "running": False}
        return {
            "running": self._running,
            "enabled": self.enabled,
            "mode": self.mode.value,
            "mode_profile": get_mode(self.mode).public(),
            "suppression_strength": self.suppression_strength,
            "config": self.config.as_dict(),
            "audio": audio,
            "backends": {
                "classifier": self.classifier.describe(),
                "separator": self.separator.describe(),
                "speaker": self.identifier.describe(),
                "name_awareness": self.name_awareness.spotter.describe(),
                "fallbacks": fallback_notes(),
            },
            "metrics": self.metrics.snapshot().as_dict(),
            "context": self.context.snapshot(),
            "person_lock": self.person_lock.snapshot(),
            "safety": self.safety.describe(),
            "waveform": self.buffers.waveform(),
        }
