import Foundation

/// Cascaded peaking EQ with band-interaction compensation. Port of
/// audio/biquad.py: zero added latency, and a request for -18 dB measures
/// -18 dB because neighbouring bands' overlap is inverted out.
public final class BandShaper {
    public static let bands: [Float] = [60, 160, 400, 1000, 2500, 6000]
    let q: Float = 0.9
    let sampleRate: Float
    private var coeffs: [[Double]]           // per band: b0 b1 b2 a1 a2
    private var z1: [Double], z2: [Double]
    public private(set) var target: [Float]
    private let decouple: [[Double]]

    public init(sampleRate: Float) {
        self.sampleRate = sampleRate
        let n = Self.bands.count
        coeffs = Self.bands.map { Self.peaking($0, 0.9, 0, sampleRate) }
        z1 = [Double](repeating: 0, count: n)
        z2 = [Double](repeating: 0, count: n)
        target = [Float](repeating: 0, count: n)
        decouple = Self.interactionInverse(sampleRate: sampleRate, q: 0.9)
    }

    static func peaking(_ f: Float, _ q: Float, _ gainDB: Float, _ sr: Float) -> [Double] {
        let a = pow(10, Double(gainDB) / 40)
        let w0 = 2 * Double.pi * Double(min(f, sr * 0.45)) / Double(sr)
        let alpha = sin(w0) / (2 * Double(q))
        let c = cos(w0)
        let a0 = 1 + alpha / a
        return [(1 + alpha * a) / a0, -2 * c / a0, (1 - alpha * a) / a0, -2 * c / a0, (1 - alpha / a) / a0]
    }

    static func responseDB(_ cs: [[Double]], at f: Float, _ sr: Float) -> Double {
        let w = 2 * Double.pi * Double(f) / Double(sr)
        var db = 0.0
        for c in cs {
            // |H(e^jw)| for one biquad.
            let (cw, sw, c2, s2) = (cos(w), sin(w), cos(2 * w), sin(2 * w))
            let nr = c[0] + c[1] * cw + c[2] * c2, ni = -(c[1] * sw + c[2] * s2)
            let dr = 1 + c[3] * cw + c[4] * c2, di = -(c[3] * sw + c[4] * s2)
            db += 10 * log10((nr * nr + ni * ni) / (dr * dr + di * di))
        }
        return db
    }

    /// Measure each band's effect on every centre and invert, so band gains
    /// can be solved for rather than guessed.
    static func interactionInverse(sampleRate sr: Float, q: Float) -> [[Double]] {
        let n = bands.count, probe: Float = 6
        var m = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for j in 0..<n {
            let cs = bands.enumerated().map { peaking($0.element, q, $0.offset == j ? probe : 0, sr) }
            for i in 0..<n { m[i][j] = responseDB(cs, at: bands[i], sr) / Double(probe) }
        }
        return invert(m) ?? (0..<n).map { i in (0..<n).map { $0 == i ? 1 : 0 } }
    }

    static func invert(_ a: [[Double]]) -> [[Double]]? {
        let n = a.count
        var m = a.enumerated().map { i, row in row + (0..<n).map { $0 == i ? 1.0 : 0.0 } }
        for col in 0..<n {
            guard let p = (col..<n).max(by: { abs(m[$0][col]) < abs(m[$1][col]) }), abs(m[p][col]) > 1e-12 else { return nil }
            m.swapAt(col, p)
            let d = m[col][col]
            for k in 0..<(2 * n) { m[col][k] /= d }
            for r in 0..<n where r != col {
                let f = m[r][col]
                for k in 0..<(2 * n) { m[r][k] -= f * m[col][k] }
            }
        }
        return m.map { Array($0[n...]) }
    }

    public func setGains(_ g: [Float]) {
        var changed = false
        for i in g.indices where abs(g[i] - target[i]) >= 0.05 { changed = true }
        guard changed else { return }
        target = g.map { clamp($0, -40, 12) }
        for i in 0..<Self.bands.count {
            var raw = 0.0
            for j in 0..<Self.bands.count { raw += decouple[i][j] * Double(target[j]) }
            coeffs[i] = Self.peaking(Self.bands[i], q, Float(max(-48, min(18, raw))), sampleRate)
        }
    }

    public var isFlat: Bool { target.allSatisfy { abs($0) < 0.05 } }

    public func response(at f: Float) -> Float { Float(Self.responseDB(coeffs, at: f, sampleRate)) }

    /// In-place DF2-transposed cascade; state carries across blocks.
    public func process(_ x: UnsafeMutablePointer<Float>, count n: Int) {
        guard !isFlat else { return }
        for b in 0..<coeffs.count {
            let c = coeffs[b]
            let (b0, b1, b2, a1, a2) = (c[0], c[1], c[2], c[3], c[4])
            var s1 = z1[b], s2 = z2[b]
            for i in 0..<n {
                let v = Double(x[i])
                let y = b0 * v + s1
                s1 = b1 * v - a1 * y + s2
                s2 = b2 * v - a2 * y
                x[i] = Float(y)
            }
            z1[b] = s1; z2[b] = s2
        }
    }

    public func reset() { for i in z1.indices { z1[i] = 0; z2[i] = 0 } }
}

/// Instant-attack, smooth-release limiter. Boosts from Person Lock, Name
/// Awareness and the safety lift can stack; clipping a siren would be worse
/// than useless.
public final class SoftLimiter {
    let threshold: Float
    let release: Float
    private var gain: Float = 1
    public private(set) var active = false

    public init(sampleRate: Float, thresholdDB: Float = -1, releaseMS: Float = 80) {
        threshold = powf(10, thresholdDB / 20)
        release = expf(-1 / (sampleRate * releaseMS / 1000))
    }

    public func process(_ x: UnsafeMutablePointer<Float>, count n: Int) {
        var peak: Float = 0
        for i in 0..<n { peak = max(peak, abs(x[i])) }
        active = peak > threshold
        let target = active ? threshold / peak : 1
        gain = target < gain ? target : target + (gain - target) * release
        for i in 0..<n { x[i] *= gain }
    }

    public func reset() { gain = 1; active = false }
}

/// Where decisions become sound. Port of audio/mixer.py.
public final class AudioMixer {
    public let sampleRate: Float
    let slewSamples: Int
    let bandSlew: Float = 0.25
    let shaper: BandShaper
    let limiter: SoftLimiter
    private var streamGain = [Float](repeating: 1, count: 4)
    public private(set) var bandDB = [Float](repeating: 0, count: BandShaper.bands.count)
    public private(set) var streamGainDB = [Float](repeating: 0, count: 4)
    private var env: [Float]
    private let weights: [[Float]]   // per category, per band

    public init(sampleRate: Float, maxBlock: Int) {
        self.sampleRate = sampleRate
        slewSamples = Int(sampleRate * 0.025)
        shaper = BandShaper(sampleRate: sampleRate)
        limiter = SoftLimiter(sampleRate: sampleRate)
        env = [Float](repeating: 0, count: maxBlock)
        weights = SoundCategory.allCases.map { AudioMixer.bandWeights($0) }
    }

    static func bandWeights(_ c: SoundCategory) -> [Float] {
        let (lo0, hi0) = c.info.bandHz
        let lo = max(lo0, 20), hi = max(hi0, lo * 1.05)
        return BandShaper.bands.map { f in
            let bLo = log10f(f / 1.6), bHi = log10f(f * 1.6)
            return clamp(max(min(log10f(hi), bHi) - max(log10f(lo), bLo), 0) / (bHi - bLo), 0, 1)
        }
    }

    public func reset() {
        shaper.reset(); limiter.reset()
        for i in 0..<4 { streamGain[i] = 1; streamGainDB[i] = 0 }
        for i in bandDB.indices { bandDB[i] = 0 }
    }

    /// Mix the separated streams into `out`. `bypass` sums them untouched.
    public func mix(_ streams: [[Float]], _ decisions: [GainDecision], bypass: Bool,
                    into out: UnsafeMutablePointer<Float>, count n: Int) {
        if bypass {
            for i in 0..<n { out[i] = streams[0][i] + streams[1][i] + streams[2][i] + streams[3][i] }
            limiter.process(out, count: n)
            return
        }
        let groups = group(decisions)
        for i in 0..<n { out[i] = 0 }
        for kind in [StreamKind.primarySpeech, .secondarySpeech, .transient] {
            let db = groups[kind.rawValue] ?? 0
            streamGainDB[kind.rawValue] = db
            let tgt = powf(10, db / 20)
            let prev = streamGain[kind.rawValue]
            let s = streams[kind.rawValue]
            let ns = min(n, slewSamples)
            for i in 0..<n {
                let g = i < ns ? prev + (tgt - prev) * Float(i) / Float(ns) : tgt
                out[i] += s[i] * g
            }
            streamGain[kind.rawValue] = tgt
        }
        // Environment: frequency-selective shaping.
        let target = environmentBands(decisions)
        for b in bandDB.indices {
            // Attenuate fast, recover slowly: a one-frame detection dropout
            // must not let the rumble surge back.
            let rate = target[b] > bandDB[b] ? bandSlew * 0.25 : bandSlew
            bandDB[b] += rate * (target[b] - bandDB[b])
        }
        shaper.setGains(bandDB)
        let e = streams[StreamKind.environment.rawValue]
        for i in 0..<n { env[i] = e[i] }
        env.withUnsafeMutableBufferPointer { shaper.process($0.baseAddress!, count: n) }
        for i in 0..<n { out[i] += env[i] }
        streamGainDB[StreamKind.environment.rawValue] = bandDB.reduce(0, +) / Float(bandDB.count)
        limiter.process(out, count: n)
    }

    /// One gain per scalar stream. Critical decisions take their stream;
    /// speech decisions govern the whole speech path unless one is promoted
    /// (see the note in mixer.py on why labels and streams differ).
    func group(_ ds: [GainDecision]) -> [Int: Float] {
        var result: [Int: Float] = [:]
        for kind in StreamKind.allCases {
            let g = ds.filter { $0.stream == kind }
            guard !g.isEmpty else { continue }
            let critical = g.filter { $0.alwaysAudible || $0.safetyOverride }
            if !critical.isEmpty { result[kind.rawValue] = critical.map(\.gainDB).max()!; continue }
            var num: Float = 0, den: Float = 0
            for d in g { let w = max(d.confidence, 1e-3); num += w * d.gainDB; den += w }
            result[kind.rawValue] = num / den
        }
        let speech = ds.filter { $0.stream == .primarySpeech || $0.stream == .secondarySpeech }
        if !speech.isEmpty {
            let promoted = speech.filter { $0.stream == .primarySpeech && $0.priority >= .high }
            if !promoted.isEmpty {
                result[StreamKind.primarySpeech.rawValue] = promoted.map(\.gainDB).max()!
            } else {
                var num: Float = 0, den: Float = 0
                for d in speech { let w = max(d.confidence, 1e-3); num += w * d.gainDB; den += w }
                result[StreamKind.primarySpeech.rawValue] = num / den
            }
        }
        return result
    }

    func environmentBands(_ ds: [GainDecision]) -> [Float] {
        let n = BandShaper.bands.count
        var num = [Float](repeating: 0, count: n), den = [Float](repeating: 0, count: n)
        var floor = [Float](repeating: -.infinity, count: n)
        for d in ds where d.stream == .environment {
            let w = weights[d.category.rawValue]
            for b in 0..<n {
                let wt = w[b] * max(d.confidence, 1e-3)
                num[b] += wt * d.gainDB; den[b] += wt
                if (d.alwaysAudible || d.safetyOverride) && w[b] > 0.05 { floor[b] = max(floor[b], d.gainDB) }
            }
        }
        // Absence of a detection is not detection of absence: hold the curve.
        guard den.contains(where: { $0 > 1e-6 }) else { return bandDB }
        return (0..<n).map { b in
            var v = den[b] > 1e-6 ? num[b] / den[b] : 0
            if floor[b].isFinite { v = max(v, floor[b]) }
            return clamp(v, -30, 12)
        }
    }
}
