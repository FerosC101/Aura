import Foundation

/// MFCC front-end at 16 kHz, shared by Person Lock and Name Awareness.
/// Port of speaker/spectral_embedder.py. Runs on the semantic worker, never on
/// the audio thread, so it is free to allocate.
public final class MFCC {
    public static let rate: Float = 16000
    let frameLen = 400, hopLen = 160, nFFT = 512, nMels = 40, nCep = 20
    private let fft = RealFFT(512)
    private let fb: [[Float]]
    private let dct: [[Float]]
    private let ham: [Float]

    public init() {
        ham = (0..<400).map { 0.54 - 0.46 * cosf(2 * .pi * Float($0) / 399) }
        fb = MFCC.melFilterbank(nFilters: 40, nFFT: 512, sr: 16000, fmin: 60, fmax: 8000)
        dct = (0..<20).map { k in
            (0..<40).map { n -> Float in
                var v = cosf(.pi * Float(k) * Float(2 * n + 1) / 80) * sqrtf(2.0 / 40)
                if k == 0 { v /= sqrtf(2) }
                return v
            }
        }
    }

    static func hzToMel(_ f: Float) -> Float { 2595 * log10f(1 + f / 700) }
    static func melToHz(_ m: Float) -> Float { 700 * (powf(10, m / 2595) - 1) }

    static func melFilterbank(nFilters: Int, nFFT: Int, sr: Float, fmin: Float, fmax: Float) -> [[Float]] {
        let nBins = nFFT / 2 + 1
        let lo = hzToMel(fmin), hi = hzToMel(fmax)
        let pts = (0..<(nFilters + 2)).map { melToHz(lo + (hi - lo) * Float($0) / Float(nFilters + 1)) }
        let binPts = pts.map { min(max(Int(floorf(Float(nFFT + 1) * $0 / sr)), 0), nBins - 1) }
        var fb = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: nFilters)
        for i in 0..<nFilters {
            let l = binPts[i]
            var c = binPts[i + 1], r = binPts[i + 2]
            if c == l { c = min(l + 1, nBins - 1) }
            if r == c { r = min(c + 1, nBins - 1) }
            for b in l..<c { fb[i][b] = Float(b - l) / Float(max(c - l, 1)) }
            for b in c..<r { fb[i][b] = Float(r - b) / Float(max(r - c, 1)) }
            let area = fb[i].reduce(0, +)
            if area > 0 { fb[i] = fb[i].map { $0 / area } }
        }
        return fb
    }

    /// Resample to 16 kHz. Integer ratios use boxcar decimation (a cheap
    /// anti-alias); anything else falls back to linear interpolation.
    public static func toRate(_ x: [Float], from sr: Float) -> [Float] {
        if sr == rate { return x }
        let ratio = sr / rate
        if ratio == ratio.rounded(), ratio >= 1 {
            let f = Int(ratio)
            return stride(from: 0, to: x.count - f + 1, by: f).map { i in
                var s: Float = 0
                for j in 0..<f { s += x[i + j] }
                return s / Float(f)
            }
        }
        let n = Int(Float(x.count) / ratio)
        return (0..<n).map { i in
            let p = Float(i) * ratio
            let a = Int(p), b = min(a + 1, x.count - 1)
            return x[a] + (x[b] - x[a]) * (p - Float(a))
        }
    }

    /// Cepstra per 10 ms frame (rows), 20 coefficients each.
    public func cepstra(_ x16: [Float]) -> [[Float]] {
        guard x16.count >= frameLen else { return [] }
        var pre = [Float](repeating: 0, count: x16.count)
        pre[0] = x16[0]
        for i in 1..<x16.count { pre[i] = x16[i] - 0.97 * x16[i - 1] }
        let nFrames = 1 + (x16.count - frameLen) / hopLen
        var out: [[Float]] = []
        out.reserveCapacity(nFrames)
        var buf = [Float](repeating: 0, count: nFFT)
        for f in 0..<nFrames {
            for i in 0..<nFFT { buf[i] = i < frameLen ? pre[f * hopLen + i] * ham[i] : 0 }
            let (re, im) = fft.forward(buf)
            let pw = zip(re, im).map { $0 * $0 + $1 * $1 }
            var logMel = [Float](repeating: 0, count: nMels)
            for m in 0..<nMels {
                var s: Float = 0
                for b in 0..<pw.count { s += pw[b] * fb[m][b] }
                logMel[m] = logf(s + 1e-10)
            }
            out.append((0..<nCep).map { k in
                var s: Float = 0
                for m in 0..<nMels { s += logMel[m] * dct[k][m] }
                return s
            })
        }
        return out
    }
}

private func colMean(_ m: [[Float]], _ from: Int = 0) -> [Float] {
    let w = m[0].count
    return (from..<w).map { c in m.reduce(0) { $0 + $1[c] } / Float(m.count) }
}

private func colStd(_ m: [[Float]], _ from: Int = 0) -> [Float] {
    let means = colMean(m, from)
    return (from..<m[0].count).map { c in
        let mu = means[c - from]
        return sqrtf(m.reduce(0) { $0 + ($1[c] - mu) * ($1[c] - mu) } / Float(m.count))
    }
}

// MARK: - Speaker embedding

/// MFCC-statistics voice embedding. See spectral_embedder.py for why the
/// cepstral means are kept (they are the identity) and why the vector is
/// standardised before normalising (without it cosine saturates near 1).
public final class SpeakerEmbedder {
    public let name = "mfcc-stats-v1"
    /// (mean, std) similarity between different speakers; measured, per backend.
    public var impostorPrior: (Float, Float) = (0.953, 0.025)
    let mfcc = MFCC()

    public init() {}

    public func embed(_ samples: [Float], sampleRate: Float) -> [Float]? {
        let x = MFCC.toRate(samples, from: sampleRate)
        guard x.count >= 8000 else { return nil }
        var c = mfcc.cepstra(x)
        guard c.count >= 4 else { return nil }
        let maxE = c.map { $0[0] }.max()!
        let kept = c.filter { $0[0] > maxE - 25 }
        if kept.count >= 4 { c = kept }
        let deltas = c.count > 1 ? zip(c.dropFirst(), c).map { a, b in zip(a, b).map(-) } : c
        let (f0m, f0s, voiced) = pitchStats(x)
        var v = colMean(c, 1) + colStd(c, 1) + colStd(deltas, 1) + [f0m, f0s, voiced]
        guard v.allSatisfy(\.isFinite) else { return nil }
        let mu = v.reduce(0, +) / Float(v.count)
        v = v.map { $0 - mu }
        let sd = sqrtf(v.reduce(0) { $0 + $1 * $1 } / Float(v.count))
        guard sd > 1e-9 else { return nil }
        v = v.map { $0 / sd }
        let norm = sqrtf(v.reduce(0) { $0 + $1 * $1 })
        return norm > 1e-9 ? v.map { $0 / norm } : nil
    }

    func pitchStats(_ x: [Float]) -> (Float, Float, Float) {
        let win = 640, hop = 320, minLag = 45, maxLag = 228
        guard x.count >= win else { return (0, 0, 0) }
        let fft = RealFFT(2048)
        var f0s: [Float] = []
        let n = 1 + (x.count - win) / hop
        var buf = [Float](repeating: 0, count: 2048)
        for i in 0..<n {
            let seg = Array(x[(i * hop)..<(i * hop + win)])
            let m = seg.reduce(0, +) / Float(win)
            var e: Float = 0
            for j in 0..<2048 { buf[j] = j < win ? seg[j] - m : 0; if j < win { e += buf[j] * buf[j] } }
            if e < 1e-9 { continue }
            let (re, im) = fft.forward(buf)
            let ac = fft.inverse(re: zip(re, im).map { $0 * $0 + $1 * $1 }, im: [Float](repeating: 0, count: re.count))
            var best = minLag, bv = -Float.infinity
            for l in minLag...maxLag where ac[l] > bv { bv = ac[l]; best = l }
            if bv / e > 0.35 { f0s.append(MFCC.rate / Float(best)) }
        }
        guard !f0s.isEmpty else { return (0, 0, 0) }
        let lf = f0s.map { logf($0) }
        let mu = lf.reduce(0, +) / Float(lf.count)
        let sd = sqrtf(lf.reduce(0) { $0 + ($1 - mu) * ($1 - mu) } / Float(lf.count))
        return (mu * 8, sd * 8, Float(f0s.count) / Float(n) * 8)
    }

    public static func similarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var d: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        guard na > 1e-12, nb > 1e-12 else { return 0 }
        return clamp(d / (sqrtf(na) * sqrtf(nb)), 0, 1)
    }
}

// MARK: - Person Lock

/// An enrolled voice. Embedding only -- there is no field that can hold audio.
public struct SpeakerProfile: Codable, Identifiable, Sendable, Equatable {
    public var id: String
    public var name: String
    public var centroid: [Float]
    public var samples: Int
    public var backend: String
    public var created: Date
}

public struct LockState: Sendable, Equatable {
    public var active = false
    public var name: String?
    public var similarity: Float = 0
    public var margin: Float = 0
    public init() {}
}

/// Cohort-normalised speaker lock. Port of speaker/person_lock.py.
public final class PersonLock {
    public static let zThreshold: Float = 1.5
    static let priorStrength: Float = 12
    public let embedder: SpeakerEmbedder
    public private(set) var profiles: [SpeakerProfile] = []
    public var selectedID: String?
    public private(set) var state = LockState()
    let sampleRate: Float
    private var buffer: [Float] = []
    private var impostor: [Float] = []
    private var holdUntil: Date = .distantPast
    /// Profiles are edited from the UI while the semantic worker reads them.
    /// The worker is not real-time, so an ordinary lock is fine here.
    private let lock = NSLock()

    public init(sampleRate: Float, embedder: SpeakerEmbedder = SpeakerEmbedder()) {
        self.sampleRate = sampleRate
        self.embedder = embedder
    }

    public var selected: SpeakerProfile? { profiles.first { $0.id == selectedID } }

    /// Build a profile from clips. The clips are consumed here; only the
    /// centroid survives this call.
    public func enrol(name: String, clips: [[Float]], sampleRate: Float, consent: Bool) throws -> SpeakerProfile {
        guard consent else { throw EnrolError.consentRequired }
        let vecs = clips.compactMap { embedder.embed($0, sampleRate: sampleRate) }
        guard !vecs.isEmpty else { throw EnrolError.noUsableSpeech }
        var c = colMean(vecs)
        let n = sqrtf(c.reduce(0) { $0 + $1 * $1 })
        c = c.map { $0 / max(n, 1e-9) }
        let p = SpeakerProfile(id: UUID().uuidString, name: name, centroid: c, samples: vecs.count,
                               backend: embedder.name, created: Date())
        lock.withLock {
            profiles.append(p)
            if selectedID == nil { selectedID = p.id }
        }
        return p
    }

    public func restore(_ ps: [SpeakerProfile], selected: String?) {
        lock.withLock {
            // An embedding from another backend is not comparable; drop it.
            profiles = ps.filter { $0.backend == embedder.name }
            selectedID = profiles.contains { $0.id == selected } ? selected : profiles.first?.id
            resetLocked()
        }
    }

    public func select(_ id: String?) { lock.withLock { selectedID = id; resetLocked() } }

    public func delete(_ id: String) {
        lock.withLock {
            profiles.removeAll { $0.id == id }
            if selectedID == id { selectedID = profiles.first?.id }
            resetLocked()
        }
    }

    public func clear() { lock.withLock { profiles = []; selectedID = nil; resetLocked() } }

    public func reset() { lock.withLock { resetLocked() } }

    private func resetLocked() { buffer = []; impostor = []; state = LockState(); holdUntil = .distantPast }

    public func update(_ block: [Float], speechPresent: Bool, now: Date = Date()) -> LockState {
        lock.lock(); defer { lock.unlock() }
        guard let target = selected else { state = LockState(); return state }
        let window = Int(1.5 * sampleRate)
        if speechPresent {
            buffer.append(contentsOf: block)
            if buffer.count > window { buffer.removeFirst(buffer.count - window) }
        }
        guard buffer.count >= window else {
            if now > holdUntil { state.active = false; state.name = target.name }
            return state
        }
        let vec = embedder.embed(buffer, sampleRate: sampleRate)
        buffer.removeFirst(window / 2)
        guard let vec else { return state }
        let score = SpeakerEmbedder.similarity(vec, target.centroid)
        var bestOther: Float = 0
        for p in profiles where p.id != target.id {
            let s = SpeakerEmbedder.similarity(vec, p.centroid)
            bestOther = max(bestOther, s)
            impostor.append(s)
        }
        let margin = self.margin(score)
        let active = margin >= Self.zThreshold && score >= embedder.impostorPrior.0 && score >= bestOther
        if active { holdUntil = now.addingTimeInterval(0.8) } else { impostor.append(score) }
        if impostor.count > 120 { impostor.removeFirst(impostor.count - 120) }
        state = LockState()
        state.active = active; state.name = target.name; state.similarity = score; state.margin = margin
        return state
    }

    func margin(_ score: Float) -> Float {
        let (pm, ps) = embedder.impostorPrior
        guard !impostor.isEmpty else { return (score - pm) / max(ps, 1e-3) }
        let n = Float(impostor.count)
        let mu = impostor.reduce(0, +) / n
        let sd = sqrtf(impostor.reduce(0) { $0 + ($1 - mu) * ($1 - mu) } / n)
        let w = n / (n + Self.priorStrength)
        return (score - (w * mu + (1 - w) * pm)) / max(w * sd + (1 - w) * ps, 1e-3)
    }

    public enum EnrolError: LocalizedError {
        case consentRequired, noUsableSpeech
        public var errorDescription: String? {
            switch self {
            case .consentRequired: "Person Lock needs this person's consent before saving their voice."
            case .noUsableSpeech: "We couldn't hear enough speech. Try again somewhere a little quieter."
            }
        }
    }
}

// MARK: - Name Awareness

/// Query-by-example spotting with DTW. Never transcribes. Port of
/// speech/name_awareness.py, including its honest scope: a name said as its
/// own short utterance, not buried mid-sentence.
public final class NameSpotter {
    public static let threshold: Float = 0.28
    let mfcc = MFCC()
    public private(set) var templates: [String: [[[Float]]]] = [:]
    let sampleRate: Float
    private var buffer: [Float] = []
    private var holdUntil = Date.distantPast
    private var lastAttempt = Date.distantPast
    public var enabled = false
    public private(set) var lastWord: String?
    private let lock = NSLock()

    public init(sampleRate: Float) { self.sampleRate = sampleRate }

    /// Templates are cepstral sequences -- derived features, not audio.
    public func restore(_ t: [String: [[[Float]]]]) { lock.withLock { templates = t } }

    func sequence(_ x: [Float], sr: Float) -> [[Float]]? {
        var c = mfcc.cepstra(MFCC.toRate(x, from: sr))
        guard c.count >= 3 else { return nil }
        let maxE = c.map { $0[0] }.max()!
        let kept = c.filter { $0[0] > maxE - 25 }
        if kept.count >= 3 { c = kept }
        let shape = c.map { Array($0[1...12]) }
        let mu = colMean(shape)
        return shape.map { zip($0, mu).map(-) }
    }

    @discardableResult
    public func enrol(_ word: String, clips: [[Float]], sampleRate sr: Float) -> Bool {
        let seqs = clips.compactMap { sequence($0, sr: sr) }
        guard !seqs.isEmpty else { return false }
        lock.withLock { templates[word] = seqs }
        return true
    }

    public func forget() { lock.withLock { templates = [:]; lastWord = nil; buffer = [] } }

    public static func dtw(_ a: [[Float]], _ b: [[Float]], band: Float = 0.34) -> Float {
        let n = a.count, m = b.count
        guard n > 0, m > 0 else { return .infinity }
        func unit(_ v: [Float]) -> [Float] { let s = sqrtf(v.reduce(0) { $0 + $1 * $1 }) + 1e-9; return v.map { $0 / s } }
        let an = a.map(unit), bn = b.map(unit)
        let width = max(2, Int(band * Float(max(n, m))))
        var acc = [[Float]](repeating: [Float](repeating: .infinity, count: m + 1), count: n + 1)
        acc[0][0] = 0
        for i in 1...n {
            let centre = i * m / n
            let lo = max(1, centre - width), hi = min(m, centre + width)
            guard lo <= hi else { continue }
            for j in lo...hi {
                var dot: Float = 0
                for k in 0..<an[i - 1].count { dot += an[i - 1][k] * bn[j - 1][k] }
                acc[i][j] = (1 - dot) + min(acc[i - 1][j], acc[i][j - 1], acc[i - 1][j - 1])
            }
        }
        return acc[n][m].isFinite ? acc[n][m] / Float(n + m) : .infinity
    }

    /// Best matching word in `x`, or nil.
    public func detect(_ x: [Float], sampleRate sr: Float) -> (String, Float)? {
        guard let seq = sequence(x, sr: sr) else { return nil }
        var best: (String, Float)?
        for (w, ts) in lock.withLock({ templates }) {
            for t in ts {
                let r = Float(seq.count) / Float(max(t.count, 1))
                guard r >= 0.4, r <= 3 else { continue }
                let d = Self.dtw(t, seq)
                if d < (best?.1 ?? .infinity) { best = (w, d) }
            }
        }
        guard let b = best, b.1 <= Self.threshold else { return nil }
        return b
    }

    public func update(_ block: [Float], speechPresent: Bool, now: Date = Date()) -> Bool {
        guard enabled, !lock.withLock({ templates.isEmpty }) else { return false }
        let window = Int(1.0 * sampleRate)
        if speechPresent {
            buffer.append(contentsOf: block)
            if buffer.count > window { buffer.removeFirst(buffer.count - window) }
        }
        if now < holdUntil { return true }
        guard buffer.count >= window, now.timeIntervalSince(lastAttempt) >= 1.5 else { return false }
        lastAttempt = now
        if let (w, _) = detect(buffer, sampleRate: sampleRate) {
            lastWord = w
            holdUntil = now.addingTimeInterval(2.5)
            buffer = []
            return true
        }
        return false
    }
}
