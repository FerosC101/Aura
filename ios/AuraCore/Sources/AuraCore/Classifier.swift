import Foundation

/// Weighted geometric blend of (score, weight) pairs. A single strongly
/// contradicting cue can veto a detection, which an arithmetic mean cannot do.
///
/// Fixed-arity overloads rather than an array parameter: detectors run on the
/// audio render thread, and an array literal per call is a heap allocation
/// fourteen times a frame.
typealias Cue = (Float, Float)

@inline(__always) private func acc(_ c: Cue, _ a: inout Float, _ w: inout Float) {
    a += c.1 * logf(max(c.0, 1e-3)); w += c.1
}

@inline(__always) private func finish(_ a: Float, _ w: Float) -> Float {
    w > 0 ? clamp(expf(a / w), 0, 1) : 0
}

private func combine(_ c0: Cue, _ c1: Cue, _ c2: Cue, _ c3: Cue, _ c4: Cue) -> Float {
    var a: Float = 0, w: Float = 0
    acc(c0, &a, &w); acc(c1, &a, &w); acc(c2, &a, &w); acc(c3, &a, &w); acc(c4, &a, &w)
    return finish(a, w)
}

private func combine(_ c0: Cue, _ c1: Cue, _ c2: Cue, _ c3: Cue, _ c4: Cue, _ c5: Cue) -> Float {
    var a: Float = 0, w: Float = 0
    acc(c0, &a, &w); acc(c1, &a, &w); acc(c2, &a, &w); acc(c3, &a, &w); acc(c4, &a, &w); acc(c5, &a, &w)
    return finish(a, w)
}

private func combine(_ c0: Cue, _ c1: Cue, _ c2: Cue, _ c3: Cue, _ c4: Cue, _ c5: Cue, _ c6: Cue) -> Float {
    var a: Float = 0, w: Float = 0
    acc(c0, &a, &w); acc(c1, &a, &w); acc(c2, &a, &w); acc(c3, &a, &w)
    acc(c4, &a, &w); acc(c5, &a, &w); acc(c6, &a, &w)
    return finish(a, w)
}

// MARK: - Detectors (tuned against measured features; see dsp_classifier.py)

enum Detectors {
    static func humanSpeech(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.modulationDepth, 0.18, 0.45), 2.2),
            (ramp(f.voicing, 0.18, 0.45), 2.0),
            (window(f.modulationRateHz, 1.2, 2.2, 8.0, 11.0), 1.0),
            (ramp(f.lowTotal, 0.06, 0.22), 1.4),
            (window(f.entropy, 0.28, 0.48, 0.86, 0.95), 1.2),
            (ramp(f.fluxVariance * 1000, 8, 45), 1.2),
            (invRamp(f.tonality, 0.50, 0.85), 1.0),
        )
    }

    static func crowdSpeech(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.voicing, 0.14, 0.38), 2.2),
            (window(f.modulationDepth, 0.0, 0.04, 0.30, 0.52), 1.8),
            (window(f.entropy, 0.32, 0.48, 0.78, 0.92), 1.2),
            (invRamp(f.flatness, 0.08, 0.28), 1.4),
            (invRamp(f.lowTotal, 0.10, 0.30), 1.6),
            (invRamp(f.tonality, 0.45, 0.80), 1.0),
        )
    }

    static func vehicleHorn(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.tonality, 0.45, 0.72), 2.2),
            (window(f.tonalPeakHz, 180, 280, 2000, 2800), 1.8),
            (invRamp(f.peakRangeHz, 130, 400), 2.0),
            (invRamp(f.entropy, 0.38, 0.72), 1.4),
            (ramp(f.dbfs, -44, -22), 1.0),
            (invRamp(f.modulationDepth, 0.18, 0.45), 1.6),
        )
    }

    static func emergencySiren(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.peakRangeHz, 140, 430), 2.6),
            (ramp(f.tonality, 0.45, 0.78), 1.8),
            (window(f.tonalPeakHz, 330, 520, 3200, 4500), 1.4),
            (invRamp(f.entropy, 0.32, 0.68), 1.2),
            (ramp(f.dbfs, -50, -28), 0.8),
        )
    }

    static func engineNoise(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.lowTotal, 0.25, 0.55), 2.5),
            (invRamp(f.centroidHz, 180, 700), 2.0),
            (invRamp(f.modulationDepth, 0.14, 0.45), 1.2),
            (ramp(f.stationarity, 0.58, 0.92), 1.0),
            (invRamp(f.zcr, 0.10, 0.34), 1.0),
            (invRamp(f.tonality, 0.55, 0.85), 0.8),
        )
    }

    static func generalTraffic(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.flatness, 0.10, 0.32), 1.8),
            (ramp(f.entropy, 0.58, 0.84), 1.6),
            (window(f.centroidHz, 380, 850, 3600, 6200), 1.4),
            (invRamp(f.voicing, 0.12, 0.38), 1.6),
            (invRamp(f.tonality, 0.35, 0.72), 1.0),
            (ramp(f.dbfs, -46, -26), 1.0),
            (invRamp(f.hfTotal, 0.18, 0.42), 0.8),
        )
    }

    static func alarm(_ f: FrameFeatures) -> Float {
        combine(
            (window(f.tonalPeakHz, 1700, 2500, 6200, 8200), 2.2),
            (ramp(f.tonality, 0.42, 0.70), 1.6),
            (ramp(f.modulationDepth, 0.14, 0.45), 1.8),
            (invRamp(f.entropy, 0.28, 0.62), 1.0),
            (invRamp(f.peakRangeHz, 140, 420), 1.0),
        )
    }

    static func warningSignal(_ f: FrameFeatures) -> Float {
        combine(
            (window(f.tonalPeakHz, 560, 780, 1750, 2500), 2.2),
            (ramp(f.tonality, 0.42, 0.72), 1.6),
            (ramp(f.modulationDepth, 0.12, 0.42), 1.6),
            (invRamp(f.entropy, 0.28, 0.62), 1.0),
            (invRamp(f.peakRangeHz, 140, 420), 1.2),
        )
    }

    static func bicycleBell(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.tonality, 0.58, 0.88), 2.0),
            (window(f.tonalPeakHz, 1400, 1900, 7000, 9200), 1.8),
            (invRamp(f.entropy, 0.20, 0.58), 1.4),
            (invRamp(f.modulationDepth, 0.22, 0.52), 1.2),
            (invRamp(f.peakRangeHz, 130, 400), 1.0),
        )
    }

    static func music(_ f: FrameFeatures) -> Float {
        combine(
            (invRamp(f.fluxVariance * 1000, 10, 45), 1.8),
            (ramp(f.tonality, 0.22, 0.52), 1.4),
            (window(f.modulationDepth, 0.0, 0.03, 0.32, 0.58), 1.4),
            (ramp(f.stationarity, 0.72, 0.95), 1.2),
            (window(f.entropy, 0.30, 0.45, 0.76, 0.90), 1.2),
            (ramp(f.lowTotal, 0.06, 0.24), 1.0),
            (ramp(f.voicing, 0.10, 0.35), 0.8),
        )
    }

    static func mechanicalNoise(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.hfTotal, 0.10, 0.34), 2.0),
            (window(f.centroidHz, 1100, 2100, 6800, 9500), 1.6),
            (ramp(f.fluxVariance * 1000, 4, 26), 1.4),
            (ramp(f.entropy, 0.52, 0.80), 1.2),
            (invRamp(f.voicing, 0.12, 0.38), 1.0),
            (ramp(f.dbfs, -52, -30), 0.8),
        )
    }

    static func announcement(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.modulationDepth, 0.26, 0.50), 2.8),
            (ramp(f.voicing, 0.22, 0.48), 1.6),
            (invRamp(f.lowTotal, 0.05, 0.20), 2.2),
            (invRamp(f.hfTotal, 0.04, 0.20), 1.2),
            (window(f.entropy, 0.18, 0.30, 0.64, 0.86), 1.0),
            (ramp(f.fluxVariance * 1000, 8, 45), 1.0),
        )
    }

    static func approachingVehicle(_ f: FrameFeatures) -> Float {
        combine(
            (ramp(f.levelSlopeDBs, 2.0, 12.0), 2.6),
            (ramp(f.lowTotal, 0.12, 0.42), 1.6),
            (ramp(f.dbfs, -52, -32), 1.0),
            (invRamp(f.modulationDepth, 0.30, 0.65), 0.6),
            (invRamp(f.voicing, 0.15, 0.40), 2.0),
        )
    }

    static func ambient(_ f: FrameFeatures) -> Float {
        combine(
            (invRamp(f.dbfs, -52, -28), 1.4),
            (ramp(f.flatness, 0.14, 0.40), 1.2),
            (invRamp(f.tonality, 0.25, 0.62), 1.0),
            (invRamp(f.voicing, 0.12, 0.35), 1.0),
            (invRamp(f.modulationDepth, 0.18, 0.45), 0.8),
        )
    }
}

/// Dependency-free multi-label classifier. Port of dsp_classifier.py.
public final class SoundClassifier {
    public static let silenceFloorDBFS: Float = -68

    public let sampleRate: Float
    public var threshold: Float = 0.42
    let attack: Float = 0.55
    let release: Float = 0.12

    private let mixFX: FeatureExtractor
    private var streamFX: [FeatureExtractor]
    private var smoothed = [Float](repeating: 0, count: SoundCategory.allCases.count)
    private var raw = [Float](repeating: 0, count: SoundCategory.allCases.count)
    private var levels = [Float](repeating: -90, count: SoundCategory.allCases.count)
    public private(set) var lastMixFeatures = FrameFeatures()

    public init(sampleRate: Float, frameLength: Int) {
        self.sampleRate = sampleRate
        mixFX = FeatureExtractor(sampleRate: sampleRate, frameLength: frameLength)
        streamFX = StreamKind.allCases.map { _ in FeatureExtractor(sampleRate: sampleRate, frameLength: frameLength) }
    }

    public func reset() {
        mixFX.reset()
        streamFX.forEach { $0.reset() }
        for i in smoothed.indices { smoothed[i] = 0 }
    }

    /// Where each category is analysed (nil = the full mix). Narrowband events
    /// stay on the mix: see the note in dsp_classifier.py on why the isolated
    /// transient stream produced siren false positives.
    private static func analysedOn(_ c: SoundCategory) -> StreamKind? {
        switch c {
        case .humanSpeech, .announcement: .primarySpeech
        case .crowdSpeech: .secondarySpeech
        case .engineNoise, .generalTraffic, .mechanicalNoise, .music, .approachingVehicle: .environment
        default: nil
        }
    }

    private static func detect(_ c: SoundCategory, _ f: FrameFeatures) -> Float {
        switch c {
        case .humanSpeech: Detectors.humanSpeech(f)
        case .crowdSpeech: Detectors.crowdSpeech(f)
        case .vehicleHorn: Detectors.vehicleHorn(f)
        case .emergencySiren: Detectors.emergencySiren(f)
        case .engineNoise: Detectors.engineNoise(f)
        case .generalTraffic: Detectors.generalTraffic(f)
        case .alarm: Detectors.alarm(f)
        case .music: Detectors.music(f)
        case .mechanicalNoise: Detectors.mechanicalNoise(f)
        case .announcement: Detectors.announcement(f)
        case .bicycleBell: Detectors.bicycleBell(f)
        case .approachingVehicle: Detectors.approachingVehicle(f)
        case .warningSignal: Detectors.warningSignal(f)
        case .ambient: Detectors.ambient(f)
        }
    }

    /// Classify one frame. `streams` is the separator output, indexed by
    /// `StreamKind.rawValue`; pass nil to classify the mix alone.
    public func classify(mix: UnsafePointer<Float>, count n: Int,
                         streams: [UnsafePointer<Float>]? = nil) -> [SoundEvent] {
        let mixF = mixFX.extract(mix, count: n)
        lastMixFeatures = mixF
        var streamF: [FrameFeatures?] = [nil, nil, nil, nil]
        if let streams {
            for k in StreamKind.allCases { streamF[k.rawValue] = streamFX[k.rawValue].extract(streams[k.rawValue], count: n) }
        }
        let silent = mixF.dbfs < Self.silenceFloorDBFS

        for c in SoundCategory.allCases {
            let on = Self.analysedOn(c)
            let f = on.flatMap { streamF[$0.rawValue] } ?? mixF
            var score: Float = 0
            if !(silent && c != .ambient) {
                score = Self.detect(c, f)
                if on != nil, streams != nil, f.dbfs < Self.silenceFloorDBFS { score = 0 }
            }
            raw[c.rawValue] = score
            levels[c.rawValue] = f.dbfs
        }
        applyExclusions()

        var events: [SoundEvent] = []
        events.reserveCapacity(8)
        for c in SoundCategory.allCases {
            let s = raw[c.rawValue]
            let prev = smoothed[c.rawValue]
            let coeff = s > prev ? attack : release
            let v = prev + coeff * (s - prev)
            smoothed[c.rawValue] = v
            if v >= threshold {
                events.append(SoundEvent(c, confidence: v, levelDB: levels[c.rawValue]))
            }
        }
        if events.isEmpty {
            events.append(SoundEvent(.ambient, confidence: max(smoothed[SoundCategory.ambient.rawValue], 0.3), levelDB: mixF.dbfs))
        }
        events.sort { $0.confidence > $1.confidence }
        return events
    }

    // MARK: - Exclusions (see _apply_exclusions in dsp_classifier.py)

    private func score(_ c: SoundCategory) -> Float { raw[c.rawValue] }
    private func scale(_ c: SoundCategory, _ f: Float) { raw[c.rawValue] *= clamp(f, 0, 1) }

    private func compete(_ group: [SoundCategory], gamma: Float) {
        guard let best = group.max(by: { score($0) < score($1) }) else { return }
        let b = score(best)
        guard b > 1e-3 else { return }
        for c in group where c != best { scale(c, powf(score(c) / b, gamma)) }
    }

    private func applyExclusions() {
        let tonal: [SoundCategory] = [.emergencySiren, .vehicleHorn, .alarm, .warningSignal, .bicycleBell]
        compete(tonal, gamma: 2.0)
        compete([.humanSpeech, .crowdSpeech, .announcement], gamma: 1.6)
        compete([.generalTraffic, .mechanicalNoise, .ambient], gamma: 1.3)
        compete([.music, .crowdSpeech], gamma: 1.5)
        compete([.music, .engineNoise], gamma: 1.5)

        let tonalStrength = tonal.map(score).max() ?? 0
        if tonalStrength > 0.45 {
            let veto = 1 - 0.85 * tonalStrength
            for c in [SoundCategory.crowdSpeech, .music, .humanSpeech, .announcement, .generalTraffic, .ambient] { scale(c, veto) }
        }
        let approaching = score(.approachingVehicle)
        if approaching > 0.45 {
            let claim = 1 - 0.85 * approaching
            scale(.engineNoise, claim)
            scale(.generalTraffic, claim)
        }
        var fg: Float = 0
        for c in SoundCategory.allCases where c != .ambient { fg = max(fg, score(c)) }
        scale(.ambient, powf(max(1 - fg, 0), 1.6))
    }
}
