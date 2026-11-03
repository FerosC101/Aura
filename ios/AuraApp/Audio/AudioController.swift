import AuraCore
import AVFoundation
import Synchronization

/// State shared with the audio render thread. Everything here is either
/// immutable, atomic, or only touched while the engine is stopped.
final class RenderBridge: @unchecked Sendable {
    enum Route: Int { case idle = 0, live, demo, recordOnly }

    let route = Atomic<Int>(Route.idle.rawValue)
    let recording = Atomic<Bool>(false)
    let recordCount = Atomic<Int>(0)
    let peakBits = Atomic<UInt32>(0)

    var processor: AuraProcessor
    // Demo signal: set only while the engine is stopped.
    var demo: [Float] = []
    var demoPos = 0
    // Recording: preallocated, filled by the sink block.
    let recordCapacity: Int
    let recordBuffer: UnsafeMutablePointer<Float>

    init(processor: AuraProcessor) {
        self.processor = processor
        recordCapacity = 48000 * 8
        recordBuffer = .allocate(capacity: recordCapacity)
        recordBuffer.initialize(repeating: 0, count: recordCapacity)
    }

    deinit { recordBuffer.deallocate() }

    var inputPeak: Float { Float(bitPattern: peakBits.load(ordering: .relaxed)) }
}

/// Owns AVAudioSession and AVAudioEngine.
///
/// Graph: microphone -> AVAudioSinkNode -> AuraProcessor -> AVAudioSourceNode
/// -> mixer -> output. Both custom nodes run on the render thread; the sink
/// pushes captured samples, the source pulls processed ones. Nothing is tapped
/// on a background queue, because taps deliver audio in ~100 ms chunks -- far
/// too late for the ear.
@MainActor
final class AudioController {
    enum Mode: Equatable { case idle, live, demo(String, Comparison), recording }

    enum AudioError: LocalizedError {
        case noMicPermission, needsHeadphones, engine(String)
        var errorDescription: String? {
            switch self {
            case .noMicPermission: "AURA needs the microphone to hear the world around you. You can allow it in Settings."
            case .needsHeadphones: "Connect headphones or earbuds first. Playing the microphone through the speaker would feed back."
            case .engine(let m): m
            }
        }
    }

    let bridge: RenderBridge
    private(set) var mode: Mode = .idle
    private var engine = AVAudioEngine()
    private var sink: AVAudioSinkNode?
    private var source: AVAudioSourceNode?
    var onRouteLost: (() -> Void)?
    var onInterrupted: (() -> Void)?
    /// Builds a processor for a device rate other than the current one. The
    /// model owns settings and voiceprints, so it re-applies them.
    var makeProcessor: ((Float) -> AuraProcessor)?

    init(processor: AuraProcessor) {
        bridge = RenderBridge(processor: processor)
        let nc = NotificationCenter.default
        nc.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] note in
            let reason = (note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt).flatMap(AVAudioSession.RouteChangeReason.init)
            MainActor.assumeIsolated {
                // Headphones pulled out mid-session: stop before the mic starts
                // feeding back through the speaker.
                if reason == .oldDeviceUnavailable, self?.mode == .live { self?.stop(); self?.onRouteLost?() }
            }
        }
        nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            let type = (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init)
            MainActor.assumeIsolated {
                if type == .began, self?.mode != .idle { self?.stop(); self?.onInterrupted?() }
            }
        }
    }

    var processor: AuraProcessor { bridge.processor }

    // MARK: - Permissions and routes

    static var micPermission: AVAudioApplication.recordPermission { AVAudioApplication.shared.recordPermission }

    static func requestMic() async -> Bool { await AVAudioApplication.requestRecordPermission() }

    /// Whether the current output is private to the listener.
    static var headphonesConnected: Bool {
        #if targetEnvironment(simulator)
        return true   // The simulator routes through the Mac; trust the developer.
        #else
        let safe: Set<AVAudioSession.Port> = [.headphones, .bluetoothA2DP, .bluetoothLE, .bluetoothHFP, .usbAudio, .lineOut]
        return AVAudioSession.sharedInstance().currentRoute.outputs.contains { safe.contains($0.portType) }
        #endif
    }

    static var outputName: String {
        AVAudioSession.sharedInstance().currentRoute.outputs.first?.portName ?? "—"
    }

    static var inputName: String {
        AVAudioSession.sharedInstance().currentRoute.inputs.first?.portName ?? "Microphone"
    }

    /// Device-side latency: input + output buffers + one IO period.
    static var deviceLatencyMS: Float {
        let s = AVAudioSession.sharedInstance()
        return Float((s.inputLatency + s.outputLatency + s.ioBufferDuration) * 1000)
    }

    // MARK: - Lifecycle

    func startLive() throws {
        guard Self.micPermission == .granted else { throw AudioError.noMicPermission }
        guard Self.headphonesConnected else { throw AudioError.needsHeadphones }
        try configure(route: .live, withInput: true)
        mode = .live
    }

    func startDemo(_ samples: [Float], key: String, comparison: Comparison) throws {
        stop()
        bridge.demo = samples
        bridge.demoPos = 0
        try configure(route: .demo, withInput: false)
        // After configure(): its internal stop() resets the comparison.
        bridge.processor.update { $0.comparison = comparison }
        mode = .demo(key, comparison)
    }

    /// Switch Raw / ANC / AURA without restarting playback.
    func setComparison(_ c: Comparison) {
        bridge.processor.update { $0.comparison = c }
        if case .demo(let k, _) = mode { mode = .demo(k, c) }
    }

    func stop() {
        engine.stop()
        if let sink { engine.detach(sink) }
        if let source { engine.detach(source) }
        sink = nil
        source = nil
        bridge.route.store(RenderBridge.Route.idle.rawValue, ordering: .relaxed)
        bridge.processor.stopWorker()
        bridge.processor.update { $0.comparison = .aura }
        mode = .idle
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Record `seconds` of microphone audio for enrolment. The samples are
    /// returned to the caller, turned into an embedding, and dropped -- they
    /// are never written anywhere.
    func record(seconds: Double) async throws -> [Float] {
        guard Self.micPermission == .granted else { throw AudioError.noMicPermission }
        // Live mode already has the microphone attached; anything else (idle,
        // or a demo, whose graph has no input) needs a capture-only graph.
        let reuseLive = mode == .live
        if !reuseLive {
            try configure(route: .recordOnly, withInput: true)
            mode = .recording
        }
        bridge.recordCount.store(0, ordering: .relaxed)
        bridge.recording.store(true, ordering: .releasing)
        try await Task.sleep(for: .seconds(seconds))
        bridge.recording.store(false, ordering: .releasing)
        let n = min(bridge.recordCount.load(ordering: .acquiring), bridge.recordCapacity)
        let clip = Array(UnsafeBufferPointer(start: bridge.recordBuffer, count: n))
        // Overwrite the capture buffer: the raw audio should not outlive this call.
        bridge.recordBuffer.update(repeating: 0, count: bridge.recordCapacity)
        if !reuseLive { stop() }
        return clip
    }

    var recordingLevel: Float { bridge.inputPeak }

    // MARK: - Graph

    private func configure(route: RenderBridge.Route, withInput: Bool) throws {
        stop()
        let session = AVAudioSession.sharedInstance()
        do {
            if withInput {
                // A2DP only: routes output to Bluetooth earbuds at full quality
                // while capturing from the phone's own microphone. Allowing HFP
                // would switch the earbuds to a 16 kHz call link.
                try session.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetoothA2DP])
            } else {
                try session.setCategory(.playback, mode: .default)
            }
            try session.setPreferredSampleRate(48000)
            try session.setPreferredIOBufferDuration(0.005)
            try session.setActive(true)
        } catch {
            throw AudioError.engine("Couldn't start audio: \(error.localizedDescription)")
        }

        engine = AVAudioEngine()
        let rate = withInput ? engine.inputNode.outputFormat(forBus: 0).sampleRate : 48000
        if abs(Double(bridge.processor.sampleRate) - rate) >= 1 {
            // Safe: the engine is stopped, so the render thread is not running.
            guard let make = makeProcessor else { throw AudioError.engine("Unsupported sample rate \(Int(rate)) Hz") }
            bridge.processor = make(Float(rate))
        }
        let processor = bridge.processor
        let mono = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1)!
        let bridge = self.bridge

        if withInput {
            let inFormat = engine.inputNode.outputFormat(forBus: 0)
            let sink = AVAudioSinkNode { _, frames, abl in
                let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
                guard let data = buffers[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
                let n = Int(frames)
                var peak: Float = 0
                for i in 0..<n { peak = max(peak, abs(data[i])) }
                bridge.peakBits.store(peak.bitPattern, ordering: .relaxed)
                if bridge.route.load(ordering: .relaxed) == RenderBridge.Route.live.rawValue {
                    bridge.processor.pushInput(data, n)
                }
                if bridge.recording.load(ordering: .acquiring) {
                    let c = bridge.recordCount.load(ordering: .relaxed)
                    let k = min(n, bridge.recordCapacity - c)
                    if k > 0 {
                        (bridge.recordBuffer + c).update(from: data, count: k)
                        bridge.recordCount.store(c + k, ordering: .releasing)
                    }
                }
                return noErr
            }
            engine.attach(sink)
            engine.connect(engine.inputNode, to: sink, format: inFormat)
            self.sink = sink
        }

        let source = AVAudioSourceNode(format: mono) { isSilence, _, frames, abl in
            let buffers = UnsafeMutableAudioBufferListPointer(abl)
            guard let out = buffers[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            let n = Int(frames)
            switch RenderBridge.Route(rawValue: bridge.route.load(ordering: .relaxed)) ?? .idle {
            case .live:
                bridge.processor.pullOutput(out, n)
            case .demo:
                // Feed the scene into the processor as if it were the mic.
                bridge.demo.withUnsafeBufferPointer { d in
                    guard let base = d.baseAddress, d.count > 0 else { return }
                    var remaining = n
                    while remaining > 0 {
                        let k = min(remaining, d.count - bridge.demoPos)
                        bridge.processor.pushInput(base + bridge.demoPos, k)
                        bridge.demoPos = (bridge.demoPos + k) % d.count
                        remaining -= k
                    }
                }
                bridge.processor.pullOutput(out, n)
            case .idle, .recordOnly:
                for i in 0..<n { out[i] = 0 }
                isSilence.pointee = true
            }
            return noErr
        }
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: mono)
        self.source = source

        processor.flush()
        bridge.route.store(route.rawValue, ordering: .relaxed)
        engine.prepare()
        do { try engine.start() } catch {
            bridge.route.store(RenderBridge.Route.idle.rawValue, ordering: .relaxed)
            throw AudioError.engine("Couldn't start audio: \(error.localizedDescription)")
        }
        processor.startWorker()
    }
}
