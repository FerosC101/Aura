import Accelerate
import Foundation

/// Acoustic descriptors for one frame. Mirrors `FrameFeatures` in features.py.
public struct FrameFeatures: Sendable {
    public var rms: Float = 0
    public var dbfs: Float = -90
    public var peak: Float = 0
    public var crestFactor: Float = 1
    public var centroidHz: Float = 0
    public var rolloffHz: Float = 0
    public var bandwidthHz: Float = 0
    public var flatness: Float = 1
    public var entropy: Float = 1
    public var slope: Float = 0
    public var zcr: Float = 0

    // Band energy ratios.
    public var sub: Float = 0, low: Float = 0, lowMid: Float = 0, mid: Float = 0
    public var highMid: Float = 0, high: Float = 0, air: Float = 0, speech: Float = 0
    public var lowTotal: Float { sub + low }
    public var hfTotal: Float { high + air }

    public var tonality: Float = 0
    public var harmonicity: Float = 0
    public var f0Hz: Float = 0
    public var voicing: Float = 0
    public var nTonalPeaks: Int = 0
    public var tonalPeakHz: Float = 0
    public var tonalPersistence: Float = 0

    public var flux: Float = 0
    public var fluxVariance: Float = 0
    public var levelSlopeDBs: Float = 0
    public var modulationDepth: Float = 0
    public var modulationRateHz: Float = 0
    public var peakDriftHz: Float = 0
    public var peakRangeHz: Float = 0
    public var pitchVariationHz: Float = 0
    public var stationarity: Float = 0
}

/// Streaming feature extractor. A line-by-line port of features.py; see that
/// file for why each measure is computed the way it is.
///
/// All working memory is allocated in `init` for one fixed frame length, so
/// `extract` can run on the audio render thread.
public final class FeatureExtractor {
    public let sampleRate: Float
    public let frameLength: Int
    static let nLogBands = 32

    private let fft: RealFFT
    private let acFFT: RealFFT
    private let win: [Float]
    private var windowed: [Float]
    private var re: [Float], im: [Float]
    private var mag: [Float], prevMag: [Float], power: [Float]
    private var floorBuf: [Float]
    private var freqs: [Float]
    private var hasPrev = false
    private var bandSlices: [(Int, Int)] = []
    private var psd: [Float]
    private var slopeX: [Float], slopeY: [Float]
    private var acIn: [Float], acRe: [Float], acIm: [Float], acOut: [Float]
    private var hop: Float

    private var fluxHist = History(capacity: 48)
    private var envHist = History(capacity: 48)
    private var dbHist = History(capacity: 48)
    private var f0Hist = History(capacity: 48)
    private var tonalHist = History(capacity: 48)
    private var peakHist = History(capacity: 48)   // NaN when no tonal peak
    private var scratch: [Float]

    // Named band boundaries, Hz.
    private let bandEdges: [(Float, Float)] = [
        (20, 80), (80, 250), (250, 800), (800, 2000), (2000, 4000), (4000, 8000), (8000, 20000),
    ]

    public init(sampleRate: Float, frameLength: Int) {
        self.sampleRate = sampleRate
        self.frameLength = frameLength
        fft = RealFFT(frameLength)
        acFFT = RealFFT(RealFFT.supportedLength(atLeast: 2 * frameLength))
        win = hanning(frameLength)
        windowed = [Float](repeating: 0, count: frameLength)
        let bins = frameLength / 2 + 1
        re = [Float](repeating: 0, count: bins)
        im = [Float](repeating: 0, count: bins)
        mag = [Float](repeating: 0, count: bins)
        prevMag = [Float](repeating: 0, count: bins)
        power = [Float](repeating: 0, count: bins)
        floorBuf = [Float](repeating: 0, count: bins)
        freqs = (0..<bins).map { Float($0) * sampleRate / Float(frameLength) }
        psd = [Float](repeating: 0, count: Self.nLogBands)
        slopeX = [Float](repeating: 0, count: bins)
        slopeY = [Float](repeating: 0, count: bins)
        acIn = [Float](repeating: 0, count: acFFT.n)
        acRe = [Float](repeating: 0, count: acFFT.bins)
        acIm = [Float](repeating: 0, count: acFFT.bins)
        acOut = [Float](repeating: 0, count: acFFT.n)
        scratch = [Float](repeating: 0, count: 64)
        hop = Float(frameLength) / sampleRate
        buildBandSlices()
    }

    public func reset() {
        hasPrev = false
        fluxHist.clear(); envHist.clear(); dbHist.clear()
        f0Hist.clear(); tonalHist.clear(); peakHist.clear()
    }

    private func buildBandSlices() {
        let lo = log(Double(50)), hi = log(Double(min(8000, sampleRate / 2 - 1)))
        let edges = (0...Self.nLogBands).map { Float(exp(lo + (hi - lo) * Double($0) / Double(Self.nLogBands))) }
        func searchLeft(_ v: Float) -> Int {
            var i = 0
            while i < freqs.count && freqs[i] < v { i += 1 }
            return i
        }
        bandSlices = (0..<Self.nLogBands).map { i in
            let a = searchLeft(edges[i])
            var b = searchLeft(edges[i + 1])
            if b <= a { b = min(a + 1, freqs.count) }
            return (a, b)
        }
    }

    // MARK: - Main entry point

    public func extract(_ x: UnsafePointer<Float>, count n: Int) -> FrameFeatures {
        precondition(n == frameLength, "FeatureExtractor built for \(frameLength), got \(n)")
        var f = FrameFeatures()

        // ---- level ----
        var sumSq: Float = 0
        vDSP_svesq(x, 1, &sumSq, vDSP_Length(n))
        f.rms = sqrtf(sumSq / Float(n) + EPS)
        var pk: Float = 0
        vDSP_maxmgv(x, 1, &pk, vDSP_Length(n))
        f.peak = pk
        f.dbfs = toDB(f.rms)
        f.crestFactor = f.peak / max(f.rms, EPS)
        var crossings = 0
        for i in 1..<n where (x[i] < 0) != (x[i - 1] < 0) { crossings += 1 }
        f.zcr = Float(crossings) / Float(n - 1)

        // ---- spectrum ----
        let bins = n / 2 + 1
        windowed.withUnsafeMutableBufferPointer { w in
            vDSP_vmul(x, 1, win, 1, w.baseAddress!, 1, vDSP_Length(n))
        }
        windowed.withUnsafeBufferPointer { w in
            re.withUnsafeMutableBufferPointer { r in
                im.withUnsafeMutableBufferPointer { i in
                    fft.forward(w.baseAddress!, re: r.baseAddress!, im: i.baseAddress!)
                }
            }
        }
        var total: Float = 0
        for k in 0..<bins {
            let p = re[k] * re[k] + im[k] * im[k]
            power[k] = p
            mag[k] = sqrtf(p)
            total += p
        }
        total += EPS

        // ---- spectral shape ----
        var cSum: Float = 0
        for k in 0..<bins { cSum += freqs[k] * power[k] }
        f.centroidHz = cSum / total
        var bSum: Float = 0
        for k in 0..<bins { let d = freqs[k] - f.centroidHz; bSum += d * d * power[k] }
        f.bandwidthHz = sqrtf(bSum / total)
        var cum: Float = 0
        let target = 0.85 * (total - EPS)
        f.rolloffHz = freqs[bins - 1]
        for k in 0..<bins {
            cum += power[k]
            if cum >= target { f.rolloffHz = freqs[k]; break }
        }
        var m = 0
        for k in 0..<bins where freqs[k] > 60 && freqs[k] < 10000 {
            slopeX[m] = log10f(freqs[k]); slopeY[m] = log10f(power[k] + EPS); m += 1
        }
        if m > 4 {
            f.slope = slopeX.withUnsafeBufferPointer { sx in
                slopeY.withUnsafeBufferPointer { sy in linearSlope(sx.baseAddress!, sy.baseAddress!, m) }
            }
        }

        // ---- log-band PSD, flatness, entropy ----
        var logSum: Double = 0, aSum: Double = 0, pSum: Double = 0
        for b in 0..<Self.nLogBands {
            let (a, e) = bandSlices[b]
            var s: Float = 0
            if e > a { for k in a..<e { s += power[k] } ; s /= Float(e - a) }
            psd[b] = s
            logSum += log(Double(s) + 1e-18)
            aSum += Double(s)
            pSum += Double(s)
        }
        let gmean = exp(logSum / Double(Self.nLogBands))
        let amean = aSum / Double(Self.nLogBands) + 1e-18
        f.flatness = clamp(Float(gmean / amean), 0, 1)
        var ent: Double = 0
        for b in 0..<Self.nLogBands {
            let p = Double(psd[b]) / (pSum + 1e-18)
            ent -= p * log(p + 1e-18)
        }
        f.entropy = Float(ent / log(Double(Self.nLogBands)))

        // ---- band energies ----
        func band(_ lo: Float, _ hi: Float) -> Float {
            var s: Float = 0
            for k in 0..<bins where freqs[k] >= lo && freqs[k] < hi { s += power[k] }
            return s / total
        }
        f.sub = band(20, 80); f.low = band(80, 250); f.lowMid = band(250, 800)
        f.mid = band(800, 2000); f.highMid = band(2000, 4000); f.high = band(4000, 8000)
        f.air = band(8000, 20000); f.speech = band(200, 3500)

        // ---- tonality ----
        tonality(bins, &f)

        // ---- harmonicity / f0 ----
        harmonicity(x, n, &f)
        f.voicing = (f.f0Hz >= 70 && f.f0Hz <= 350) ? f.harmonicity : 0

        // ---- recurrent dynamics ----
        if hasPrev {
            var pos: Float = 0, meanMag: Float = 0, dot: Float = 0, n1: Float = 0, n2: Float = 0
            for k in 0..<bins {
                let d = mag[k] - prevMag[k]
                if d > 0 { pos += d }
                meanMag += mag[k]
                dot += mag[k] * prevMag[k]
                n1 += mag[k] * mag[k]
                n2 += prevMag[k] * prevMag[k]
            }
            meanMag /= Float(bins)
            f.flux = pos / (Float(bins) * (meanMag + EPS))
            f.stationarity = dot / (sqrtf(n1) * sqrtf(n2) + EPS)
        }
        for k in 0..<bins { prevMag[k] = mag[k] }
        hasPrev = true

        fluxHist.push(f.flux)
        envHist.push(f.rms)
        dbHist.push(f.dbfs)
        f0Hist.push(f.f0Hz)
        tonalHist.push(f.tonality)
        peakHist.push(f.tonality > 0.45 ? f.tonalPeakHz : .nan)

        if fluxHist.count > 4 { f.fluxVariance = fluxHist.variance() }
        (f.modulationDepth, f.modulationRateHz) = modulation()
        f.levelSlopeDBs = levelSlope()
        f.pitchVariationHz = pitchVariation()
        (f.peakDriftHz, f.peakRangeHz) = peakTrack()
        if tonalHist.count > 0 {
            var c = 0
            for i in 0..<tonalHist.count where tonalHist.at(i) > 0.5 { c += 1 }
            f.tonalPersistence = Float(c) / Float(tonalHist.count)
        }
        return f
    }

    // MARK: - Internals

    private func tonality(_ bins: Int, _ f: inout FrameFeatures) {
        let binHz = freqs[1] - freqs[0]
        var k = max(9, Int((600 / binHz).rounded())) | 1
        k = min(k, (bins / 2) | 1)
        guard k >= 5 else { return }
        let c = k / 2
        let notch = max(1, Int((120 / binHz).rounded()))
        mag.withUnsafeBufferPointer { mp in
            floorBuf.withUnsafeMutableBufferPointer { fp in
                notchedFloor(mp.baseAddress!, count: bins, halfWidth: c, notch: notch, into: fp.baseAddress!)
            }
        }
        var maxMag: Float = 0, argMax = -1
        for i in 0..<bins where freqs[i] >= 150 && freqs[i] <= 8000 {
            if mag[i] > maxMag { maxMag = mag[i]; argMax = i }
        }
        guard argMax >= 0 else { return }
        let gate = 0.2 * maxMag + EPS
        var peaks = 0
        var maxRatio: Float = 1
        var anyGate = false
        for i in 0..<bins where freqs[i] >= 150 && freqs[i] <= 8000 && mag[i] > gate {
            let r = mag[i] / (floorBuf[i] + EPS)
            if r > 8 { peaks += 1 }
            if !anyGate || r > maxRatio { maxRatio = r }
            anyGate = true
        }
        if !anyGate { maxRatio = 1 }
        f.tonality = clamp(log10f(max(maxRatio, 1)) / 2, 0, 1)
        f.nTonalPeaks = peaks
        f.tonalPeakHz = freqs[argMax]
    }

    private func harmonicity(_ x: UnsafePointer<Float>, _ n: Int, _ f: inout FrameFeatures) {
        guard n >= 256 else { return }
        let sr = Int(sampleRate)
        let minLag = max(2, sr / 500)
        let maxLag = min(n - 1, sr / 60)
        guard maxLag > minLag else { return }
        var mean: Float = 0
        vDSP_meanv(x, 1, &mean, vDSP_Length(n))
        var energy: Float = 0
        for i in 0..<n { let v = x[i] - mean; acIn[i] = v; energy += v * v }
        energy += EPS
        for i in n..<acFFT.n { acIn[i] = 0 }
        acIn.withUnsafeBufferPointer { ip in
            acRe.withUnsafeMutableBufferPointer { r in
                acIm.withUnsafeMutableBufferPointer { i in
                    acFFT.forward(ip.baseAddress!, re: r.baseAddress!, im: i.baseAddress!)
                }
            }
        }
        for k in 0..<acFFT.bins { acRe[k] = acRe[k] * acRe[k] + acIm[k] * acIm[k]; acIm[k] = 0 }
        acRe.withUnsafeBufferPointer { r in
            acIm.withUnsafeBufferPointer { i in
                acOut.withUnsafeMutableBufferPointer { o in
                    acFFT.inverse(re: r.baseAddress!, im: i.baseAddress!, into: o.baseAddress!)
                }
            }
        }
        var best = minLag
        var bestV = -Float.infinity
        for lag in minLag...maxLag where acOut[lag] > bestV { bestV = acOut[lag]; best = lag }
        let strength = clamp(bestV / energy, 0, 1)
        if best >= maxLag - 1 || best <= minLag + 1 {
            f.harmonicity = strength * 0.4
            f.f0Hz = 0
            return
        }
        f.harmonicity = strength
        f.f0Hz = strength > 0.25 ? sampleRate / Float(best) : 0
    }

    private func modulation() -> (Float, Float) {
        let n = envHist.count
        guard n >= 12 else { return (0, 0) }
        let mean = envHist.mean()
        guard mean >= 1e-6 else { return (0, 0) }
        let frameRate = 1 / max(hop, 1e-4)
        // Naive DFT: at most 48 points, and vDSP needs friendlier lengths.
        for i in 0..<n {
            let w: Float = n > 1 ? 0.5 - 0.5 * cosf(2 * .pi * Float(i) / Float(n - 1)) : 1
            scratch[i] = (envHist.at(i) - mean) * w
        }
        var bestMag: Float = -1, bestHz: Float = 0, total: Float = 0
        for k in 0...(n / 2) {
            var sr: Float = 0, si: Float = 0
            for t in 0..<n {
                let a = -2 * Float.pi * Float(k * t) / Float(n)
                sr += scratch[t] * cosf(a); si += scratch[t] * sinf(a)
            }
            let mk = sqrtf(sr * sr + si * si)
            total += mk
            let hz = Float(k) * frameRate / Float(n)
            if hz >= 1.5 && hz <= 10 && mk > bestMag { bestMag = mk; bestHz = hz }
        }
        guard bestMag >= 0, total >= EPS else { return (0, 0) }
        let depth = clamp(2 * bestMag / (Float(n) * mean + EPS), 0, 1)
        return (depth, bestHz)
    }

    private func levelSlope() -> Float {
        let n = dbHist.count
        guard n >= 8 else { return 0 }
        for i in 0..<n { scratch[i] = Float(i) * hop }
        var y64 = (Double(0), Double(0), Double(0), Double(0))
        for i in 0..<n {
            let a = Double(scratch[i]), b = Double(dbHist.at(i))
            y64.0 += a; y64.1 += b; y64.2 += a * a; y64.3 += a * b
        }
        let dn = Double(n)
        let den = dn * y64.2 - y64.0 * y64.0
        guard abs(den) > 1e-12 else { return 0 }
        return clamp(Float((dn * y64.3 - y64.0 * y64.1) / den), -200, 200)
    }

    private func pitchVariation() -> Float {
        var c = 0, s: Float = 0
        for i in 0..<f0Hist.count where f0Hist.at(i) > 0 { c += 1; s += f0Hist.at(i) }
        guard c >= 6 else { return 0 }
        let m = s / Float(c)
        var v: Float = 0
        for i in 0..<f0Hist.count where f0Hist.at(i) > 0 { let d = f0Hist.at(i) - m; v += d * d }
        return sqrtf(v / Float(c))
    }

    private func peakTrack() -> (Float, Float) {
        var c = 0, s: Float = 0, lo = Float.infinity, hi = -Float.infinity
        for i in 0..<peakHist.count {
            let v = peakHist.at(i)
            if v.isNaN { continue }
            c += 1; s += v; lo = min(lo, v); hi = max(hi, v)
        }
        guard c >= 5 else { return (0, 0) }
        let m = s / Float(c)
        var v: Float = 0
        for i in 0..<peakHist.count {
            let p = peakHist.at(i)
            if !p.isNaN { v += (p - m) * (p - m) }
        }
        return (sqrtf(v / Float(c)), hi - lo)
    }
}
