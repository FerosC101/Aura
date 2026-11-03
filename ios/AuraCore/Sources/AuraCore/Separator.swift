import Accelerate
import Foundation

/// Real-time spectral source separator. Port of separation/spectral.py.
///
/// Splits each hop into four streams that sum back to the (one-hop delayed)
/// input: primary speech, secondary speech, transient events, environment.
/// Square-root Hann windows at 50% overlap satisfy COLA, so with unity masks
/// the output is the input -- "AURA off" is genuinely transparent.
///
/// Order matters, and was arrived at empirically in the Python reference:
/// transients are extracted first (else sirens land in the speech stream),
/// voiced harmonics are shielded from that extraction, and the speech mask is
/// gated on voicing (else road noise leaks into the speech stream).
public final class SourceSeparator {
    public let sampleRate: Float
    public let hop: Int
    public let nfft: Int
    public var algorithmicDelaySamples: Int { hop }

    static let speechLo: Float = 150, speechHi: Float = 5500
    let noiseAdapt: Float = 0.012
    let maskFloor: Float = 0.06

    private let fft: RealFFT
    private let window: [Float]
    private let freqs: [Float]
    private let bandWeight: [Float]
    private let bins: Int

    private var input: [Float]
    private var frame: [Float]
    private var re: [Float], im: [Float], power: [Float], mag: [Float]
    private var noisePSD: [Float], prevGain: [Float], prevSpeechPSD: [Float]
    private var comb: [Float], transientSel: [Float], speechGain: [Float], floorBuf: [Float]
    private var masks: [[Float]]               // indexed by StreamKind.rawValue
    private var mRe: [Float], mIm: [Float], outFrame: [Float]
    private var tails: [[Float]]
    public private(set) var outputs: [[Float]] // the four streams for the last hop

    private var f0: Float = 0
    private var voicingState: Float = 0
    private let acFFT: RealFFT
    private var acIn: [Float], acRe: [Float], acIm: [Float], acOut: [Float]

    public init(sampleRate: Float, hop: Int) {
        self.sampleRate = sampleRate
        self.hop = hop
        nfft = hop * 2
        fft = RealFFT(nfft)
        bins = nfft / 2 + 1
        // Periodic Hann; its square root is COLA-compliant at 50% overlap.
        window = (0..<nfft).map { sqrtf(0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(hop * 2))) }
        freqs = (0..<(hop + 1)).map { Float($0) * sampleRate / Float(hop * 2) }
        let taper: Float = 250, lo = SourceSeparator.speechLo, hi = SourceSeparator.speechHi
        bandWeight = freqs.map { f in
            if f < lo - taper || f > hi + taper { return 0 }
            if f < lo { return 0.5 * (1 - cosf(.pi * (f - (lo - taper)) / taper)) }
            if f > hi { return 0.5 * (1 + cosf(.pi * (f - hi) / taper)) }
            return 1
        }
        let z = { (n: Int) in [Float](repeating: 0, count: n) }
        let nb = hop + 1, nf = hop * 2
        input = z(nf); frame = z(nf)
        re = z(nb); im = z(nb); power = z(nb); mag = z(nb)
        noisePSD = [Float](repeating: 1e-7, count: nb)
        prevGain = z(nb); prevSpeechPSD = z(nb)
        comb = z(nb); transientSel = z(nb); speechGain = z(nb); floorBuf = z(nb)
        masks = StreamKind.allCases.map { _ in z(nb) }
        mRe = z(nb); mIm = z(nb); outFrame = z(nf)
        tails = StreamKind.allCases.map { _ in z(hop) }
        outputs = StreamKind.allCases.map { _ in z(hop) }
        acFFT = RealFFT(RealFFT.supportedLength(atLeast: 2 * hop))
        acIn = z(acFFT.n); acRe = z(acFFT.bins); acIm = z(acFFT.bins); acOut = z(acFFT.n)
    }

    public func reset() {
        for i in 0..<nfft { input[i] = 0 }
        for i in 0..<bins { noisePSD[i] = 1e-7; prevGain[i] = 0; prevSpeechPSD[i] = 0 }
        for k in tails.indices { for i in 0..<hop { tails[k][i] = 0 } }
        f0 = 0; voicingState = 0
    }

    /// Mask of the last hop for `kind`, for diagnostics and benchmarks.
    public func mask(_ kind: StreamKind) -> [Float] { masks[kind.rawValue] }

    // MARK: - Main entry point

    public func separate(_ block: UnsafePointer<Float>) {
        // Shift in the new hop and analyse the full window.
        for i in 0..<hop { input[i] = input[hop + i]; input[hop + i] = block[i] }
        for i in 0..<nfft { frame[i] = input[i] * window[i] }
        frame.withUnsafeBufferPointer { fp in
            re.withUnsafeMutableBufferPointer { r in
                im.withUnsafeMutableBufferPointer { m in
                    fft.forward(fp.baseAddress!, re: r.baseAddress!, im: m.baseAddress!)
                }
            }
        }
        for k in 0..<bins { power[k] = re[k] * re[k] + im[k] * im[k]; mag[k] = sqrtf(power[k]) }

        let voicing = pitchAnalysis(block)
        transientSelector(shieldVoice: voicing > 0.45)
        speechMask()
        let vf = voicingFactor(voicing)

        for k in 0..<bins {
            let t = transientSel[k]
            let rest = 1 - t
            let sg = speechGain[k] * rest * vf
            masks[StreamKind.primarySpeech.rawValue][k] = sg * comb[k]
            masks[StreamKind.secondarySpeech.rawValue][k] = sg * (1 - comb[k])
            masks[StreamKind.transient.rawValue][k] = t
            masks[StreamKind.environment.rawValue][k] = rest - sg
        }

        for kind in StreamKind.allCases { synthesise(kind) }
    }

    private func synthesise(_ kind: StreamKind) {
        let m = kind.rawValue
        for k in 0..<bins { mRe[k] = re[k] * masks[m][k]; mIm[k] = im[k] * masks[m][k] }
        mRe.withUnsafeBufferPointer { r in
            mIm.withUnsafeBufferPointer { i in
                outFrame.withUnsafeMutableBufferPointer { o in
                    fft.inverse(re: r.baseAddress!, im: i.baseAddress!, into: o.baseAddress!)
                }
            }
        }
        for i in 0..<nfft { outFrame[i] *= window[i] }
        for i in 0..<hop {
            outputs[m][i] = outFrame[i] + tails[m][i]
            tails[m][i] = outFrame[hop + i]
        }
    }

    // MARK: - Masks

    private func speechMask() {
        for k in 0..<bins {
            let p = power[k]
            var n = noisePSD[k]
            // Recursive minimum tracking: down fast, up slowly.
            n = p < n ? 0.7 * n + 0.3 * p : n * (1 + noiseAdapt) + noiseAdapt * p
            n = max(n, 1e-9)
            noisePSD[k] = n
            let post = p / (n + 1e-12)
            let prior = 0.92 * (prevSpeechPSD[k] / (n + 1e-12)) + 0.08 * max(post - 1, 0)
            var g = prior / (1 + prior) * bandWeight[k]
            g = 0.65 * g + 0.35 * prevGain[k]
            g = min(max(g, maskFloor * bandWeight[k]), 1)
            prevGain[k] = g
            prevSpeechPSD[k] = g * g * p
            speechGain[k] = g
        }
    }

    private func voicingFactor(_ voicing: Float) -> Float {
        let target = clamp((voicing - 0.15) / 0.45, 0, 1)
        let coeff: Float = target > voicingState ? 0.5 : 0.04
        voicingState += coeff * (target - voicingState)
        return 0.12 + 0.88 * voicingState
    }

    /// Returns voicing strength and fills `comb` with the harmonic comb.
    private func pitchAnalysis(_ x: UnsafePointer<Float>) -> Float {
        let est = estimateF0(x)
        if est > 0 { f0 = f0 == 0 ? est : 0.7 * f0 + 0.3 * est } else { f0 *= 0.85 }
        guard f0 >= 60 else {
            for k in 0..<bins { comb[k] = 1 }
            return 0
        }
        let width = max(f0 * 0.35, (freqs[1] - freqs[0]) * 1.5)
        var total: Float = 1e-12, low: Float = 0
        for k in 0..<bins {
            let h = max((freqs[k] / f0).rounded(), 1)
            let d = abs(freqs[k] - h * f0) / width
            comb[k] = min(expf(-0.5 * d * d), 1)
            total += power[k]
            if abs(freqs[k] - f0) < width || abs(freqs[k] - 2 * f0) < width { low += power[k] }
        }
        let inRange: Float = (f0 >= 70 && f0 <= 350) ? 1 : 0
        return inRange * clamp((low / total) / 0.05, 0, 1)
    }

    private func estimateF0(_ x: UnsafePointer<Float>) -> Float {
        let n = hop
        let sr = Int(sampleRate)
        let minLag = max(2, sr / 350), maxLag = min(n - 1, sr / 70)
        guard maxLag > minLag + 1 else { return 0 }
        var mean: Float = 0
        vDSP_meanv(x, 1, &mean, vDSP_Length(n))
        var energy: Float = 1e-12
        for i in 0..<n { let v = x[i] - mean; acIn[i] = v; energy += v * v }
        for i in n..<acFFT.n { acIn[i] = 0 }
        acIn.withUnsafeBufferPointer { ip in
            acRe.withUnsafeMutableBufferPointer { r in
                acIm.withUnsafeMutableBufferPointer { m in acFFT.forward(ip.baseAddress!, re: r.baseAddress!, im: m.baseAddress!) }
            }
        }
        for k in 0..<acFFT.bins { acRe[k] = acRe[k] * acRe[k] + acIm[k] * acIm[k]; acIm[k] = 0 }
        acRe.withUnsafeBufferPointer { r in
            acIm.withUnsafeBufferPointer { m in
                acOut.withUnsafeMutableBufferPointer { o in acFFT.inverse(re: r.baseAddress!, im: m.baseAddress!, into: o.baseAddress!) }
            }
        }
        var best = minLag, bestV = -Float.infinity
        for lag in minLag...maxLag where acOut[lag] > bestV { bestV = acOut[lag]; best = lag }
        return bestV / energy < 0.30 ? 0 : sampleRate / Float(best)
    }

    private func transientSelector(shieldVoice: Bool) {
        let binHz = freqs[1] - freqs[0]
        var k = max(9, Int((700 / binHz).rounded())) | 1
        k = min(k, (bins / 2) | 1)
        guard k >= 5 else { for i in 0..<bins { transientSel[i] = 0 }; return }
        let notch = max(1, Int((140 / binHz).rounded()))
        mag.withUnsafeBufferPointer { mp in
            floorBuf.withUnsafeMutableBufferPointer { fp in
                notchedFloor(mp.baseAddress!, count: bins, halfWidth: k / 2, notch: notch, into: fp.baseAddress!)
            }
        }
        for i in 0..<bins {
            // Below 200 Hz "peaks" are engine harmonics, which must stay
            // suppressible rather than be protected.
            if freqs[i] < 200 { transientSel[i] = 0; continue }
            var s = clamp((mag[i] / (floorBuf[i] + 1e-12) - 3) / 7, 0, 1)
            if shieldVoice { s *= 1 - comb[i] }
            transientSel[i] = s
        }
    }
}
