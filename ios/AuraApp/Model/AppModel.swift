import AuraCore
import Foundation
import Observation
import SwiftUI

/// The app's single source of truth.
///
/// Owns the persisted profile and the audio controller, pushes settings into
/// the engine, and polls the engine's scene snapshot for the UI. The engine
/// itself never calls back into SwiftUI: the audio thread must not wait on the
/// main thread for anything.
@Observable
@MainActor
final class AppModel {
    enum Listening: Equatable { case off, live, demo(String) }

    var profile: StoredProfile { didSet { persistSoon() } }
    var scene = SceneSnapshot()
    var listening: Listening = .off
    var comparison: Comparison = .aura
    var alert: String?
    var toast: String?
    var session = SessionStats()
    /// Bumped whenever a mixer row's answer could change.
    private(set) var mixRevision = 0

    let audio: AudioController
    private let rowPolicy = PolicyEngine()   // for the UI; the audio thread owns its own
    private var poll: Timer?
    private var saveTask: Task<Void, Never>?

    init() {
        let stored = ProfileStore.load()
        profile = stored
        audio = AudioController(processor: AppModel.makeProcessor(48000, stored))
        audio.makeProcessor = { [weak self] rate in
            MainActor.assumeIsolated {
                let proc = AppModel.makeProcessor(rate, self?.profile ?? StoredProfile())
                if let self {
                    let ctx = self.policyContext(), safety = self.profile.safetySet
                    proc.update { $0.context = ctx; $0.safetyEnabled = safety }
                }
                return proc
            }
        }
        audio.onRouteLost = { [weak self] in
            self?.listening = .off
            self?.alert = "Your headphones disconnected, so AURA paused to avoid feedback."
        }
        audio.onInterrupted = { [weak self] in self?.listening = .off }
        pushSettings()
    }

    static func makeProcessor(_ rate: Float, _ p: StoredProfile) -> AuraProcessor {
        let proc = AuraProcessor(sampleRate: rate)
        proc.personLock.restore(p.speakers, selected: p.selectedSpeaker)
        proc.nameSpotter.restore(p.nameTemplates)
        proc.nameSpotter.enabled = p.nameAwareness
        return proc
    }

    var processor: AuraProcessor { audio.processor }

    // MARK: - Settings -> engine

    var mode: ListeningMode { profile.mode }

    var enabled = true { didSet { pushSettings() } }

    func policyContext(for mode: ListeningMode? = nil) -> PolicyContext {
        var c = PolicyContext()
        c.mode = mode ?? profile.mode
        c.enabled = enabled
        c.userOverrides = profile.overrides(for: c.mode)
        c.learnedBias = profile.learner.biases(c.mode)
        c.suppressionStrength = profile.suppression
        c.sceneLevelDB = -30
        return c
    }

    private func pushSettings() {
        let ctx = policyContext()
        let safety = profile.safetySet
        processor.update { $0.context = ctx; $0.safetyEnabled = safety }
        rowPolicy.safety.enabled = safety
        mixRevision += 1
    }

    func setMode(_ m: ListeningMode) {
        guard m != profile.mode else { return }
        withAnimation(.smooth(duration: 0.35)) { profile.mode = m }
        pushSettings()
    }

    // MARK: - The sound mixer

    /// What a category would get right now, before any live detection.
    func decision(_ c: SoundCategory) -> GainDecision {
        var ctx = policyContext()
        let lock = processor.semanticState.0
        ctx.personLockActive = lock.active
        ctx.personLockName = lock.name
        return rowPolicy.explain(c, ctx)
    }

    /// The categories each mode puts on its control card, in the order the
    /// brand shows them: what you want to hear first, then what to quiet.
    func controlCategories(_ m: ListeningMode) -> [SoundCategory] {
        switch m {
        case .care: [.humanSpeech, .alarm, .emergencySiren, .crowdSpeech, .music, .mechanicalNoise, .ambient]
        case .focus: [.humanSpeech, .announcement, .alarm, .crowdSpeech, .mechanicalNoise, .generalTraffic, .music]
        case .commute: [.humanSpeech, .announcement, .vehicleHorn, .emergencySiren, .crowdSpeech, .engineNoise, .generalTraffic]
        case .social: [.humanSpeech, .emergencySiren, .crowdSpeech, .music, .generalTraffic, .engineNoise]
        case .walk: [.approachingVehicle, .vehicleHorn, .emergencySiren, .bicycleBell, .humanSpeech, .generalTraffic, .crowdSpeech]
        case .custom: SoundCategory.allCases.filter { $0 != .ambient }
        }
    }

    /// Set a slider. The *correction* -- what the user chose minus what AURA
    /// would have done -- is what the preference learner learns from.
    func setGain(_ c: SoundCategory, _ db: Float?) {
        if let db {
            var base = policyContext()
            base.userOverrides[c] = nil
            let baseline = rowPolicy.explain(c, base).gainDB
            profile.learner.observe(profile.mode, c, correctionDB: db - baseline)
            if c.info.safetyCritical && db < SafetyGuard.minGainDB && profile.safetyEnabled.contains(c.info.key) {
                flash("\(c.info.label) will always stay audible")
            }
        }
        var o = profile.overrides[profile.mode.rawValue] ?? [:]
        o[c.info.key] = db
        profile.overrides[profile.mode.rawValue] = o
        pushSettings()
    }

    /// The + / − button: flip between letting a sound through and turning it down.
    func toggle(_ c: SoundCategory) {
        let d = decision(c)
        setGain(c, d.gainDB >= -2 ? -12 : 4)
    }

    func setSuppression(_ v: Float) { profile.suppression = v; pushSettings() }

    func setSafety(_ c: SoundCategory, on: Bool) {
        if on { if !profile.safetyEnabled.contains(c.info.key) { profile.safetyEnabled.append(c.info.key) } }
        else { profile.safetyEnabled.removeAll { $0 == c.info.key } }
        pushSettings()
    }

    // MARK: - Listening

    func startLive() async {
        if AudioController.micPermission == .undetermined { _ = await AudioController.requestMic() }
        do {
            try audio.startLive()
            listening = .live
            startPolling()
        } catch {
            alert = error.localizedDescription
        }
    }

    func startDemo(_ scene: DemoScene, comparison c: Comparison? = nil) {
        guard let samples = DemoCatalog.samples(scene.key) else { alert = "That demo scene is missing from the app."; return }
        if let c { comparison = c }
        setMode(scene.listeningMode)
        do {
            try audio.startDemo(samples, key: scene.key, comparison: comparison)
            listening = .demo(scene.key)
            startPolling()
        } catch {
            alert = error.localizedDescription
        }
    }

    func setComparison(_ c: Comparison) {
        comparison = c
        audio.setComparison(c)
    }

    func stop() {
        audio.stop()
        listening = .off
        poll?.invalidate()
        poll = nil
        scene = SceneSnapshot()
    }

    private func startPolling() {
        poll?.invalidate()
        poll = Timer.scheduledTimer(withTimeInterval: 1.0 / 12, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
    }

    private func tick() {
        let s = processor.snapshot()
        if s != scene { scene = s }
        if profile.sessionInsights { session.record(s, dt: 1.0 / 12) }
    }

    var totalLatencyMS: Float {
        AudioController.deviceLatencyMS + processor.hopMS + processor.algorithmicDelayMS + scene.processingMS
    }

    func isHearing(_ c: SoundCategory) -> Bool { scene.sounds.contains { $0.category == c } }

    /// Detected sounds in a fixed order -- safety first, then taxonomy order --
    /// so rows stay put while their values update.
    var liveSounds: [SoundCard] {
        scene.sounds.filter { $0.category != .ambient }.sorted {
            ($0.category.info.safetyCritical ? 0 : 1, $0.category.rawValue) < ($1.category.info.safetyCritical ? 0 : 1, $1.category.rawValue)
        }
    }

    // MARK: - Person Lock

    var speakers: [SpeakerProfile] { profile.speakers }

    func enrolSpeaker(name: String, clips: [[Float]]) throws {
        let p = try processor.personLock.enrol(name: name, clips: clips, sampleRate: processor.sampleRate, consent: true)
        profile.speakers.append(p)
        profile.selectedSpeaker = processor.personLock.selectedID
    }

    func selectSpeaker(_ id: String?) {
        processor.personLock.select(id)
        profile.selectedSpeaker = id
    }

    func deleteSpeaker(_ id: String) {
        processor.personLock.delete(id)
        profile.speakers.removeAll { $0.id == id }
        profile.selectedSpeaker = processor.personLock.selectedID
    }

    var lockedName: String? { scene.lock.active ? scene.lock.name : nil }

    // MARK: - Name Awareness

    func enrolName(_ word: String, clips: [[Float]]) -> Bool {
        guard processor.nameSpotter.enrol(word, clips: clips, sampleRate: processor.sampleRate) else { return false }
        profile.nameTemplates = processor.nameSpotter.templates
        return true
    }

    func setNameAwareness(_ on: Bool) {
        profile.nameAwareness = on
        processor.nameSpotter.enabled = on
        if !on {
            processor.nameSpotter.forget()
            profile.nameTemplates = [:]
        }
    }

    // MARK: - Privacy

    func deleteEverything() {
        stop()
        processor.personLock.clear()
        processor.nameSpotter.forget()
        let onboarded = profile.onboarded
        profile = StoredProfile()
        profile.onboarded = onboarded
        ProfileStore.delete()
        session = SessionStats()
        pushSettings()
        flash("Everything AURA stored has been deleted")
    }

    func forgetLearning() {
        _ = profile.learner.forgetAll()
        pushSettings()
    }

    /// Debug launch arguments (see LaunchOptions). No effect in release.
    func applyLaunchOptions() {
        if let n = LaunchOptions.name { profile.userName = n; if LaunchOptions.onboardingStep == nil { profile.onboarded = true } }
        if let m = LaunchOptions.mode.flatMap(ListeningMode.init(rawValue:)) { setMode(m) }
        if !LaunchOptions.voices.isEmpty { enrolDemoVoices(LaunchOptions.voices) }
        if let key = LaunchOptions.demo, let s = DemoCatalog.scenes.first(where: { $0.key == key }) { startDemo(s) }
    }

    /// Enrol names from the speech inside a bundled demo scene.
    ///
    /// Screenshot support only. It runs the ordinary `PersonLock.enrol` path on
    /// real audio -- the segment of a scene where its `human_speech` layer is
    /// sounding -- so what appears is a genuine voiceprint, not a fixture
    /// pretending to be one. Different scenes give different speakers.
    private func enrolDemoVoices(_ names: [String]) {
        let sources = ["clinic", "cafe", "classroom", "psc_demo"]
        for (i, name) in names.enumerated() {
            guard i < sources.count,
                  let scene = DemoCatalog.scenes.first(where: { $0.key == sources[i] }),
                  let speech = scene.layers.first(where: { $0.category == "human_speech" }),
                  let samples = DemoCatalog.samples(scene.key) else { continue }
            let rate = Float(48000)
            let from = min(Int(speech.start * Double(rate)), samples.count)
            let to = min(Int(speech.end * Double(rate)), samples.count)
            guard to - from > Int(rate) else { continue }
            let clip = Array(samples[from..<to])
            if let p = try? processor.personLock.enrol(name: name, clips: [clip],
                                                       sampleRate: rate, consent: true) {
                profile.speakers.append(p)
                if i == 0 { selectSpeaker(p.id) }
            }
        }
    }

    // MARK: - Utilities

    func flash(_ message: String) {
        toast = message
        Task { try? await Task.sleep(for: .seconds(2.4)); if toast == message { toast = nil } }
    }

    private func persistSoon() {
        saveTask?.cancel()
        let snapshot = profile
        saveTask = Task {
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            ProfileStore.save(snapshot)
        }
    }
}

/// In-memory, per-launch summary for the Insights tab. Counts, not content:
/// nothing about what was said, and nothing written to disk.
struct SessionStats: Equatable {
    var seconds: Double = 0
    var kept: [SoundCategory: Double] = [:]
    var quieted: [SoundCategory: Double] = [:]
    var safetyMoments = 0
    private var lastSafety = false

    mutating func record(_ s: SceneSnapshot, dt: Double) {
        guard s.frames > 0 else { return }
        seconds += dt
        for c in s.sounds where c.category != .ambient {
            if c.gainDB < -2 { quieted[c.category, default: 0] += dt } else { kept[c.category, default: 0] += dt }
        }
        if s.safetyOverride && !lastSafety { safetyMoments += 1 }
        lastSafety = s.safetyOverride
    }

    var keptSeconds: Double { kept.values.reduce(0, +) }
    var quietedSeconds: Double { quieted.values.reduce(0, +) }
}
