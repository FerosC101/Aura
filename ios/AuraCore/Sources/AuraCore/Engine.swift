import Foundation
import os
import Synchronization

/// One sound as the UI shows it.
public struct SoundCard: Sendable, Identifiable, Equatable {
    public var id: Int { category.rawValue }
    public var category: SoundCategory
    public var confidence: Float
    public var action: SoundAction
    public var gainDB: Float
    public var safetyOverride: Bool
    public var reason: String
}

/// Everything the UI needs from one moment of processing.
public struct SceneSnapshot: Sendable, Equatable {
    public var sounds: [SoundCard] = []
    public var safetyOverride = false
    public var inputDB: Float = -90
    public var outputDB: Float = -90
    public var bandDB: [Float] = [Float](repeating: 0, count: BandShaper.bands.count)
    public var lock = LockState()
    public var nameHeard = false
    public var waveIn: [Float] = []
    public var waveOut: [Float] = []
    public var separationMS: Float = 0
    public var classificationMS: Float = 0
    public var policyMS: Float = 0
    public var mixingMS: Float = 0
    public var processingMS: Float = 0
    public var load: Float = 0
    public var frames = 0
    public var dropouts = 0
    public init() {}
}

/// How the input reaches the listener. Raw and ANC bypass AURA so the three
/// can be compared honestly on the same material.
public enum Comparison: String, CaseIterable, Sendable, Identifiable {
    case raw, anc, aura
    public var id: String { rawValue }
}

/// The on-device AURA Engine. Port of pipeline/engine.py.
///
/// Threads:
/// * the **audio render thread** calls `pushInput` / `pullOutput`. It runs
///   separation, classification, policy and mixing inline -- about a
///   millisecond against a 20 ms hop -- and never blocks: settings are read
///   with a try-lock and a cached copy, speech is handed off through a
///   lock-free ring.
/// * the **semantic worker** (`startWorker`) runs Person Lock and Name
///   Awareness on 1-1.5 s windows. Those would blow the audio deadline, and
///   who is speaking does not change every 20 ms.
/// * the **main thread** changes settings and reads `snapshot()`.
public final class AuraProcessor: @unchecked Sendable {
    public let sampleRate: Float
    public let hop: Int

    public let separator: SourceSeparator
    public let classifier: SoundClassifier
    public let policy = PolicyEngine()
    public let mixer: AudioMixer
    public let personLock: PersonLock
    public let nameSpotter: NameSpotter
    let anc: ANCSimulator

    // Main thread -> audio thread.
    private let settingsLock = OSAllocatedUnfairLock(initialState: Settings())
    private var cached = Settings()
    public struct Settings: Sendable {
        public var context = PolicyContext()
        public var comparison = Comparison.aura
        /// Safety categories the user kept on. Routed through settings so
        /// the audio thread owns the policy engine outright.
        public var safetyEnabled: Set<SoundCategory> = Set(SoundCategory.safetyCritical)
    }

    // Audio thread -> main thread.
    private let snapLock = OSAllocatedUnfairLock(initialState: SceneSnapshot())
    // Worker -> audio thread.
    private let semanticLock = OSAllocatedUnfairLock(initialState: (LockState(), false))
    private var semantic = (LockState(), false)

    private let inRing: SPSCRing, outRing: SPSCRing
    private let speechRing: SPSCRing
    private let speechFlags = Atomic<Int>(0)   // hops of speech queued, sign = present
    private var inHop: [Float], outHop: [Float], ancHop: [Float], speechHop: [Float]

    // Metrics (audio thread only).
    private var stage = (sep: Float(0), cls: Float(0), pol: Float(0), mix: Float(0), total: Float(0))
    private var waveIn: [Float], waveOut: [Float]
    private var waveHead = 0
    private var frames = 0
    private var dropouts = 0
    private var lastEvents: [SoundEvent] = []
    private var lastDecisions: [GainDecision] = []

    private var worker: Thread?
    private let workerStop = Atomic<Bool>(false)

    public static let waveformPoints = 90

    public init(sampleRate: Float = 48000, hopSeconds: Float = 0.02) {
        self.sampleRate = sampleRate
        hop = Self.hopLength(for: sampleRate, seconds: hopSeconds)
        separator = SourceSeparator(sampleRate: sampleRate, hop: hop)
        classifier = SoundClassifier(sampleRate: sampleRate, frameLength: hop)
        mixer = AudioMixer(sampleRate: sampleRate, maxBlock: hop)
        personLock = PersonLock(sampleRate: sampleRate)
        nameSpotter = NameSpotter(sampleRate: sampleRate)
        anc = ANCSimulator(sampleRate: sampleRate)
        inRing = SPSCRing(capacity: hop * 16)
        outRing = SPSCRing(capacity: hop * 16)
        speechRing = SPSCRing(capacity: Int(sampleRate) * 4)
        inHop = [Float](repeating: 0, count: hop)
        outHop = [Float](repeating: 0, count: hop)
        ancHop = [Float](repeating: 0, count: hop)
        speechHop = [Float](repeating: 0, count: hop)
        waveIn = [Float](repeating: 0, count: Self.waveformPoints)
        waveOut = [Float](repeating: 0, count: Self.waveformPoints)
        outRing.writeZeros(hop)   // one hop of slack so output never starves
    }

    // MARK: - Settings (main thread)

    public func update(_ change: @Sendable (inout Settings) -> Void) {
        settingsLock.withLock { change(&$0) }
    }

    public var settings: Settings { settingsLock.withLock { $0 } }

    public var algorithmicDelayMS: Float { 1000 * Float(separator.algorithmicDelaySamples) / sampleRate }
    public var hopMS: Float { 1000 * Float(hop) / sampleRate }

    // MARK: - Streaming (audio render thread)

    /// Feed captured samples. Real-time safe.
    public func pushInput(_ x: UnsafePointer<Float>, _ n: Int) {
        if !inRing.write(x, n) { dropouts += 1 }
    }

    /// Produce `n` output samples, processing as many hops as are ready.
    public func pullOutput(_ y: UnsafeMutablePointer<Float>, _ n: Int) {
        while inRing.available >= hop {
            inHop.withUnsafeMutableBufferPointer { _ = inRing.read(into: $0.baseAddress!, hop) }
            inHop.withUnsafeBufferPointer { i in
                outHop.withUnsafeMutableBufferPointer { o in processHop(i.baseAddress!, o.baseAddress!) }
            }
            outHop.withUnsafeBufferPointer { if !outRing.write($0.baseAddress!, hop) { dropouts += 1 } }
        }
        let got = outRing.read(into: y, n)
        if got < n {
            for i in got..<n { y[i] = 0 }
            if frames > 2 { dropouts += 1 }
        }
    }

    /// Discard buffered audio, e.g. when switching between mic and demo.
    public func flush() {
        inRing.clear(); outRing.clear(); outRing.writeZeros(hop)
        separator.reset(); classifier.reset(); mixer.reset(); anc.reset()
    }

    // MARK: - One hop

    public func processHop(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        if let s = settingsLock.withLockIfAvailable({ $0 }) { cached = s }
        if policy.safety.enabled != cached.safetyEnabled { policy.safety.enabled = cached.safetyEnabled }
        if let s = semanticLock.withLockIfAvailable({ $0 }) { semantic = s }

        var src = x
        if cached.comparison == .anc {
            ancHop.withUnsafeMutableBufferPointer { a in
                anc.process(x, into: a.baseAddress!, count: hop)
                src = UnsafePointer(a.baseAddress!)
            }
        }
        let bypass = cached.comparison != .aura || !cached.context.enabled

        separator.separate(src)
        let t1 = DispatchTime.now().uptimeNanoseconds

        let events: [SoundEvent] = separator.outputs[0].withUnsafeBufferPointer { p0 in
            separator.outputs[1].withUnsafeBufferPointer { p1 in
                separator.outputs[2].withUnsafeBufferPointer { p2 in
                    separator.outputs[3].withUnsafeBufferPointer { p3 in
                        classifier.classify(mix: src, count: hop,
                                            streams: [p0.baseAddress!, p1.baseAddress!, p2.baseAddress!, p3.baseAddress!])
                    }
                }
            }
        }
        let t2 = DispatchTime.now().uptimeNanoseconds

        handOffSpeech()
        var ctx = cached.context
        ctx.personLockActive = semantic.0.active
        ctx.personLockName = semantic.0.name
        ctx.nameDetected = semantic.1
        let inDB = levelDB(src, hop)
        ctx.sceneLevelDB = inDB
        if bypass { ctx.enabled = false }
        let decisions = policy.decide(events, ctx)
        let t3 = DispatchTime.now().uptimeNanoseconds

        mixer.mix(separator.outputs, decisions, bypass: bypass, into: y, count: hop)
        let t4 = DispatchTime.now().uptimeNanoseconds

        // Metrics: exponential means, so the readout is stable but current.
        let ms = { (a: UInt64, b: UInt64) in Float(b - a) / 1e6 }
        let k: Float = 0.05
        stage.sep += k * (ms(t0, t1) - stage.sep)
        stage.cls += k * (ms(t1, t2) - stage.cls)
        stage.pol += k * (ms(t2, t3) - stage.pol)
        stage.mix += k * (ms(t3, t4) - stage.mix)
        stage.total += k * (ms(t0, t4) - stage.total)

        var pi: Float = 0, po: Float = 0
        for i in 0..<hop { pi = max(pi, abs(src[i])); po = max(po, abs(y[i])) }
        waveIn[waveHead] = pi; waveOut[waveHead] = po
        waveHead = (waveHead + 1) % Self.waveformPoints
        frames += 1
        lastEvents = events
        lastDecisions = decisions

        // Publish roughly 12 times a second.
        if frames % 4 == 0 { publish(inDB: inDB, outDB: levelDB(UnsafePointer(y), hop)) }
    }

    private func levelDB(_ x: UnsafePointer<Float>, _ n: Int) -> Float {
        var s: Float = 0
        for i in 0..<n { s += x[i] * x[i] }
        return toDB(sqrtf(s / Float(n)))
    }

    private func handOffSpeech() {
        let sp = separator.outputs[StreamKind.primarySpeech.rawValue]
        var e: Float = 0
        for v in sp { e += v * v }
        let present = toDB(sqrtf(e / Float(hop))) > -52
        guard present else { return }
        sp.withUnsafeBufferPointer { _ = speechRing.write($0.baseAddress!, hop) }
    }

    private func publish(inDB: Float, outDB: Float) {
        let byCat = Dictionary(lastDecisions.map { ($0.category, $0) }, uniquingKeysWith: { a, _ in a })
        var snap = SceneSnapshot()
        snap.sounds = lastEvents.prefix(6).map { e in
            let d = byCat[e.category]
            return SoundCard(category: e.category, confidence: e.confidence, action: d?.action ?? .preserve,
                             gainDB: d?.gainDB ?? 0, safetyOverride: d?.safetyOverride ?? false, reason: d?.reason ?? "")
        }
        snap.safetyOverride = policy.lastSafetyOverride && cached.comparison == .aura
        snap.inputDB = inDB; snap.outputDB = outDB
        snap.bandDB = mixer.bandDB
        snap.lock = semantic.0; snap.nameHeard = semantic.1
        snap.waveIn = (0..<Self.waveformPoints).map { waveIn[(waveHead + $0) % Self.waveformPoints] }
        snap.waveOut = (0..<Self.waveformPoints).map { waveOut[(waveHead + $0) % Self.waveformPoints] }
        snap.separationMS = stage.sep; snap.classificationMS = stage.cls
        snap.policyMS = stage.pol; snap.mixingMS = stage.mix
        snap.processingMS = stage.total
        snap.load = stage.total / hopMS
        snap.frames = frames; snap.dropouts = dropouts
        let published = snap
        _ = snapLock.withLockIfAvailable { $0 = published }
    }

    public func snapshot() -> SceneSnapshot { snapLock.withLock { $0 } }

    // MARK: - Semantic worker

    public func startWorker() {
        guard worker == nil else { return }
        workerStop.store(false, ordering: .relaxed)
        let t = Thread { [weak self] in
            while let self, !self.workerStop.load(ordering: .relaxed) {
                if !self.drainSemantic() { Thread.sleep(forTimeInterval: 0.02) }
            }
        }
        t.qualityOfService = .userInitiated
        t.name = "aura.semantic"
        worker = t
        t.start()
    }

    public func stopWorker() {
        workerStop.store(true, ordering: .relaxed)
        worker = nil
    }

    /// Process queued speech. Returns whether there was anything to do.
    @discardableResult
    public func drainSemantic() -> Bool {
        var did = false
        while speechRing.available >= hop {
            speechHop.withUnsafeMutableBufferPointer { _ = speechRing.read(into: $0.baseAddress!, hop) }
            let lock = personLock.update(speechHop, speechPresent: true)
            let heard = nameSpotter.update(speechHop, speechPresent: true)
            semanticLock.withLock { $0 = (lock, heard) }
            did = true
        }
        return did
    }

    /// Current Person Lock / Name Awareness state, for the UI.
    public var semanticState: (LockState, Bool) { semanticLock.withLock { $0 } }

    /// The hop nearest `seconds` that vDSP can transform: f * 2^k with f in
    /// {1, 3, 5, 15}. 48 kHz gives exactly 960 (20 ms); 44.1 kHz gives 960
    /// (21.8 ms) rather than an untransformable 882.
    public static func hopLength(for sampleRate: Float, seconds: Float = 0.02) -> Int {
        let want = Int(sampleRate * seconds)
        var best = 960, bestErr = Int.max
        for f in [1, 3, 5, 15] {
            var n = f * 16
            while n < want * 2 {
                if abs(n - want) < bestErr { bestErr = abs(n - want); best = n }
                n *= 2
            }
        }
        return best
    }
}

/// Conventional ANC, for the comparison: attenuates by frequency with no idea
/// what anything means -- the siren goes quiet along with the engine.
/// Port of demo/scenes.py:simulate_anc.
final class ANCSimulator {
    private let b: [Float], a: [Float]
    private var z1: Float = 0, z2: Float = 0
    private let gain: Float = powf(10, -18.0 / 20)
    private let highGain: Float = powf(powf(10, -18.0 / 20), 0.45)

    init(sampleRate: Float, cutoff: Float = 900) {
        // 2nd-order Butterworth low-pass (bilinear).
        let k = tanf(.pi * cutoff / sampleRate)
        let norm = 1 / (1 + sqrtf(2) * k + k * k)
        b = [k * k * norm, 2 * k * k * norm, k * k * norm]
        a = [2 * (k * k - 1) * norm, (1 - sqrtf(2) * k + k * k) * norm]
    }

    func reset() { z1 = 0; z2 = 0 }

    func process(_ x: UnsafePointer<Float>, into y: UnsafeMutablePointer<Float>, count n: Int) {
        for i in 0..<n {
            let v = x[i]
            let low = b[0] * v + z1
            z1 = b[1] * v - a[0] * low + z2
            z2 = b[2] * v - a[1] * low
            let high = v - low
            y[i] = (low * gain + high * highGain) * 0.9
        }
    }
}
