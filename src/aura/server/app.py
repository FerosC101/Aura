"""FastAPI control surface and live event stream.

The browser is a control surface, never part of the audio path. Audio is
captured, processed and played by the Python engine through CoreAudio /
PortAudio; this server only carries commands in and acoustic-scene events out.

That separation is the reason the UI is a web app at all: routing live audio
through a browser would add tens of milliseconds of WebAudio and resampling
latency to a pipeline whose entire processing budget is about one millisecond.

Live classifications go over a WebSocket. REST covers everything stateful:
profile, preferences, modes, devices, Person Lock and the Privacy Center.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..audio.devices import AUDIO_AVAILABLE, AUDIO_IMPORT_ERROR, list_devices
from ..classification.taxonomy import all_categories
from ..config import EngineConfig
from ..context.engine import MotionHint
from ..demo.scenes import COMPARISON_MODES, all_scenes, get_scene, simulate_anc
from ..demo.synth import speech, spoken_word
from ..pipeline.engine import AuraEngine
from ..policy.modes import all_modes
from ..types import AcousticScene, ListeningMode
from .schemas import (
    AudioPayload,
    CategoryGainRequest,
    ContextRequest,
    DemoStartRequest,
    EnabledRequest,
    EnrolSpeakerRequest,
    ModeRequest,
    MotionRequest,
    NameEnrolRequest,
    NameToggleRequest,
    OnboardingRequest,
    PrivacyRequest,
    SelectSpeakerRequest,
    StartEngineRequest,
    SuppressionRequest,
)

#: Synthetic voices offered for demo enrolment, so Person Lock can be shown
#: without a microphone.
DEMO_VOICES: dict[str, dict] = {
    "vince": dict(f0=112.0, formants=(500.0, 1480.0, 2500.0, 3400.0)),
    "maria": dict(f0=208.0, formants=(650.0, 1900.0, 2800.0, 3600.0)),
    "ben": dict(f0=148.0, formants=(430.0, 1100.0, 2300.0, 3300.0)),
    "ana": dict(f0=178.0, formants=(600.0, 1700.0, 2650.0, 3500.0)),
}


class ConnectionManager:
    """Fan-out of acoustic-scene events to connected clients.

    Events arrive on the audio thread and are handed to the event loop with
    ``call_soon_threadsafe``. Slow clients are dropped from the broadcast rather
    than awaited: a stalled WebSocket must never be able to reach back and
    stall the audio callback.
    """

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def _send(self, payload: dict) -> None:
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_threadsafe(self, payload: dict) -> None:
        """Called from the audio thread. Never blocks it."""
        if self._loop is None or not self.active:
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._send(payload))
            )


def _payload_to_array(p: AudioPayload) -> np.ndarray:
    arr = np.asarray(p.samples, dtype=np.float32)
    if arr.size == 0:
        raise HTTPException(400, "empty audio payload")
    return np.clip(arr, -1.0, 1.0)


def create_app(config: EngineConfig | None = None) -> FastAPI:
    engine = AuraEngine(config or EngineConfig())
    manager = ConnectionManager()
    app = FastAPI(
        title="AURA Engine",
        version="0.1.0",
        description="Hear what matters. Silence what doesn't.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine
    app.state.manager = manager
    # Rendering a demo scene costs a second or two; keep them once rendered.
    app.state.scene_cache = {}

    def on_scene(_scene: AcousticScene) -> None:
        manager.broadcast_threadsafe({"event": "scene", "data": engine.scene_dict()})

    engine.subscribe(on_scene)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def _startup() -> None:
        manager.bind_loop(asyncio.get_running_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine.close()

    # ------------------------------------------------------------------
    # Status and reference data
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "running": engine.running, "audio_available": AUDIO_AVAILABLE}

    @app.get("/api/status")
    async def status() -> dict:
        return engine.status()

    @app.get("/api/scene")
    async def scene() -> dict:
        return engine.scene_dict()

    @app.get("/api/devices")
    async def devices() -> dict:
        return {
            "available": AUDIO_AVAILABLE,
            "error": AUDIO_IMPORT_ERROR,
            "devices": [d.as_dict() for d in list_devices()],
        }

    @app.get("/api/modes")
    async def modes() -> dict:
        return {"modes": [m.public() for m in all_modes()], "current": engine.mode.value}

    @app.get("/api/categories")
    async def categories() -> dict:
        return {
            "categories": [
                {
                    "key": c.key,
                    "label": c.label,
                    "description": c.description,
                    "safety_critical": c.safety_critical,
                    "stream": c.stream.value,
                    "band_hz": list(c.band_hz),
                    "icon": c.icon,
                }
                for c in all_categories()
            ]
        }

    # ------------------------------------------------------------------
    # Engine control
    # ------------------------------------------------------------------

    @app.post("/api/engine/start")
    async def engine_start(req: StartEngineRequest = Body(default=StartEngineRequest())) -> dict:
        if engine.running:
            return engine.status()
        if req.input_device is not None:
            engine.config.input_device = req.input_device
        if req.output_device is not None:
            engine.config.output_device = req.output_device

        signal = None
        if req.demo_scene:
            signal = _render_scene(app, req.demo_scene, engine.config.sample_rate)
            engine.set_mode(get_scene(req.demo_scene).mode)
        try:
            engine.start(signal=signal)
        except Exception as exc:
            raise HTTPException(503, f"could not start audio: {exc}") from exc
        return engine.status()

    @app.post("/api/engine/stop")
    async def engine_stop() -> dict:
        engine.stop()
        return engine.status()

    @app.post("/api/engine/enabled")
    async def set_enabled(req: EnabledRequest) -> dict:
        return {"enabled": engine.set_enabled(req.enabled)}

    @app.post("/api/mode")
    async def set_mode(req: ModeRequest) -> dict:
        try:
            engine.set_mode(req.mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"mode": engine.mode.value, "mixer": engine.mixer_state()}

    @app.post("/api/suppression")
    async def set_suppression(req: SuppressionRequest) -> dict:
        return {"strength": engine.set_suppression_strength(req.strength)}

    # ------------------------------------------------------------------
    # Sound mixer
    # ------------------------------------------------------------------

    @app.get("/api/mixer")
    async def get_mixer() -> dict:
        return engine.mixer_state()

    @app.post("/api/mixer")
    async def set_mixer(req: CategoryGainRequest) -> dict:
        return engine.set_category_gain(req.category, req.gain_db, learn=req.learn)

    @app.get("/api/learning")
    async def learning() -> dict:
        return {"enabled": engine.learner.enabled, "preferences": engine.learner.summary()}

    @app.delete("/api/learning")
    async def forget_learning() -> dict:
        n = engine.learner.forget()
        engine.save_profile()
        return {"forgotten": n}

    # ------------------------------------------------------------------
    # Person Lock
    # ------------------------------------------------------------------

    @app.get("/api/speakers")
    async def speakers() -> dict:
        return engine.person_lock.snapshot()

    @app.post("/api/speakers")
    async def enrol_speaker(req: EnrolSpeakerRequest) -> dict:
        if not req.consent:
            raise HTTPException(
                403, "speaker enrolment requires explicit consent (consent=true)"
            )
        sr = engine.config.sample_rate
        if req.demo_voice:
            cfg = DEMO_VOICES.get(req.demo_voice.lower())
            if cfg is None:
                raise HTTPException(400, f"unknown demo voice {req.demo_voice!r}")
            clips = [speech(2.5, sr, seed=40 + 13 * i, syllable_rate=3.4 + 0.8 * i, **cfg)
                     for i in range(2)]
        else:
            if not req.clips:
                raise HTTPException(400, "provide clips or a demo_voice")
            clips = [_payload_to_array(c) for c in req.clips]
            sr = req.clips[0].sample_rate
        try:
            profile = engine.enrol_speaker(req.name, clips, sr, consent=True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"profile": profile.public(), "person_lock": engine.person_lock.snapshot()}

    @app.post("/api/speakers/select")
    async def select_speaker(req: SelectSpeakerRequest) -> dict:
        if not engine.select_speaker(req.profile_id):
            raise HTTPException(404, "no such speaker profile")
        return engine.person_lock.snapshot()

    @app.delete("/api/speakers/{profile_id}")
    async def delete_speaker(profile_id: str) -> dict:
        if not engine.delete_speaker(profile_id):
            raise HTTPException(404, "no such speaker profile")
        return engine.person_lock.snapshot()

    # ------------------------------------------------------------------
    # Name Awareness
    # ------------------------------------------------------------------

    @app.post("/api/name/enrol")
    async def enrol_name(req: NameEnrolRequest) -> dict:
        if not engine.profile.privacy.allow_name_awareness:
            raise HTTPException(403, "Name Awareness is off; enable it in the Privacy Center")
        sr = engine.config.sample_rate
        if req.demo_word:
            clips = [spoken_word(req.demo_word, 0.7, sr, f0=118.0, seed=s, rate_jitter=0.04)
                     for s in (5, 6)]
        else:
            if not req.clips:
                raise HTTPException(400, "provide clips or a demo_word")
            clips = [_payload_to_array(c) for c in req.clips]
            sr = req.clips[0].sample_rate
        if not engine.enrol_name(req.word, clips, sr):
            raise HTTPException(400, "could not build a template from that audio")
        return engine.name_awareness.snapshot()

    @app.post("/api/name/enabled")
    async def toggle_name(req: NameToggleRequest) -> dict:
        return {"enabled": engine.set_name_awareness(req.enabled)}

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    @app.post("/api/context")
    async def set_context(req: ContextRequest) -> dict:
        try:
            return engine.set_context(req.state)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/context/motion")
    async def motion(req: MotionRequest) -> dict:
        accepted = engine.submit_motion(
            MotionHint(
                speed_mps=req.speed_mps,
                step_rate_hz=req.step_rate_hz,
                place_category=req.place_category,
            )
        )
        if not accepted:
            raise HTTPException(403, "motion context is off; enable it in the Privacy Center")
        return engine.context.snapshot()

    # ------------------------------------------------------------------
    # Profile, onboarding and the Privacy Center
    # ------------------------------------------------------------------

    @app.get("/api/profile")
    async def profile() -> dict:
        return engine.profile.public()

    @app.post("/api/onboarding")
    async def onboarding(req: OnboardingRequest) -> dict:
        p = engine.profile
        p.user_name = req.user_name
        p.primary_uses = req.primary_uses
        p.safety_categories = req.safety_categories or engine.safety.protected_categories()
        engine.safety.enabled_categories = set(p.safety_categories)
        p.preferred_mode = ListeningMode(req.preferred_mode)
        p.onboarded = req.completed
        engine.set_mode(p.preferred_mode)
        engine.save_profile()
        return p.public()

    @app.get("/api/privacy")
    async def privacy() -> dict:
        return {
            "settings": engine.profile.privacy.as_dict(),
            "processes": [
                "Environmental audio, analysed locally, frame by frame.",
                "Sound categories and confidences, held in memory only.",
                "Speaker similarity scores, when Person Lock is on.",
            ],
            "stores": [
                "Your listening preferences and mixer positions.",
                "Learned preference adjustments.",
                "Speaker embeddings -- only with consent, and never raw audio.",
                "Trigger words for Name Awareness.",
            ],
            "never_stores": [
                "Raw or encoded microphone audio.",
                "Transcripts of anything said around you.",
                "A log of what was heard, or where.",
            ],
            "profile_path": str(engine.store.path),
            "speaker_profiles": [p.public() for p in engine.person_lock.profiles.values()],
            "cloud_inference": False,
            "audio_retained": False,
        }

    @app.patch("/api/privacy")
    async def set_privacy(req: PrivacyRequest) -> dict:
        p = engine.profile.privacy
        for field, value in req.model_dump(exclude_none=True).items():
            setattr(p, field, value)
        # Applying a privacy switch must take effect immediately, not on restart.
        engine.context.auto_enabled = p.auto_context_detection
        engine.context.allow_motion = p.allow_motion_context
        if not p.allow_name_awareness:
            engine.set_name_awareness(False)
        if not p.store_speaker_embeddings:
            engine.profile.speaker_profiles = []
        engine.save_profile()
        return engine.profile.privacy.as_dict()

    @app.delete("/api/privacy/data")
    async def delete_data() -> dict:
        speakers = engine.person_lock.clear()
        learned = engine.learner.forget()
        engine.name_awareness.spotter.forget()
        engine.profile.speaker_profiles = []
        engine.profile.trigger_words = []
        engine.profile.mode_overrides = {}
        engine.profile.preserve_counts = {}
        engine.profile.suppress_counts = {}
        deleted = engine.store.delete()
        return {
            "speaker_profiles_deleted": speakers,
            "learned_preferences_deleted": learned,
            "profile_file_deleted": deleted,
        }

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------

    @app.get("/api/demo/scenes")
    async def demo_scenes() -> dict:
        return {
            "scenes": [s.public() for s in all_scenes()],
            "comparisons": list(COMPARISON_MODES.values()),
        }

    @app.post("/api/demo/start")
    async def demo_start(req: DemoStartRequest) -> dict:
        try:
            scene = get_scene(req.scene)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

        sr = engine.config.sample_rate
        signal = _render_scene(app, req.scene, sr)
        if req.comparison == "anc":
            signal = simulate_anc(signal, sr)
        engine.stop()
        engine.set_mode(scene.mode)
        # "raw" and "anc" bypass AURA so the three can be compared honestly:
        # the same source material through three different processing chains.
        engine.set_enabled(req.comparison == "aura")
        engine.start(signal=signal, realtime=True)
        return {
            "scene": scene.public(),
            "comparison": req.comparison,
            "status": engine.status(),
        }

    # ------------------------------------------------------------------
    # Live events
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            await ws.send_json({"event": "status", "data": engine.status()})
            while True:
                # The client may send commands, but the channel exists mainly to
                # push scenes; receiving also gives us disconnect detection.
                msg = await ws.receive_json()
                if msg.get("action") == "status":
                    await ws.send_json({"event": "status", "data": engine.status()})
                elif msg.get("action") == "mode":
                    engine.set_mode(msg["mode"])
                    await ws.send_json({"event": "status", "data": engine.status()})
        except WebSocketDisconnect:
            manager.disconnect(ws)
        except Exception:
            manager.disconnect(ws)

    # ------------------------------------------------------------------
    # Static UI
    # ------------------------------------------------------------------

    ui_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    if ui_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=ui_dist / "assets"), name="assets")

        @app.get("/")
        async def index() -> Any:
            return FileResponse(ui_dist / "index.html")

        @app.get("/{path:path}")
        async def spa(path: str) -> Any:
            target = ui_dist / path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(ui_dist / "index.html")
    else:
        @app.get("/")
        async def no_ui() -> Any:
            return JSONResponse({
                "message": "AURA Engine is running. The UI has not been built.",
                "build": "cd ui && npm install && npm run build",
                "api": "/docs",
            })

    return app


def _render_scene(app: FastAPI, key: str, sample_rate: int) -> np.ndarray:
    cache = app.state.scene_cache
    ck = (key, sample_rate)
    if ck not in cache:
        cache[ck] = get_scene(key).render(sample_rate)
    return cache[ck]
