import Accelerate
import Foundation

/// Real FFT with NumPy semantics.
///
/// `forward` returns the same `n/2 + 1` complex bins as `np.fft.rfft`, and
/// `inverse` is the exact inverse, matching `np.fft.irfft`. vDSP's packed
/// real-DFT format (DC and Nyquist sharing one complex slot) and its factor-of-two
/// scaling are handled here and nowhere else, so the rest of the engine can be a
/// line-by-line port of the Python reference.
///
/// Lengths must be `f * 2^k` with `f` in {1, 3, 5, 15}: 960 (20 ms at 48 kHz),
/// 1920 and every power of two all qualify.
public final class RealFFT {
    public let n: Int
    public let bins: Int
    private let fwd: vDSP_DFT_Setup
    private let inv: vDSP_DFT_Setup
    private var inReal: [Float]
    private var inImag: [Float]
    private var outReal: [Float]
    private var outImag: [Float]

    public init(_ n: Int) {
        precondition(n % 2 == 0, "RealFFT needs an even length")
        guard let f = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(n), .FORWARD),
              let i = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(n), .INVERSE) else {
            fatalError("vDSP cannot build a real DFT of length \(n)")
        }
        self.n = n
        bins = n / 2 + 1
        fwd = f
        inv = i
        inReal = [Float](repeating: 0, count: n / 2)
        inImag = [Float](repeating: 0, count: n / 2)
        outReal = [Float](repeating: 0, count: n / 2)
        outImag = [Float](repeating: 0, count: n / 2)
    }

    deinit {
        vDSP_DFT_DestroySetup(fwd)
        vDSP_DFT_DestroySetup(inv)
    }

    /// Forward transform of `x` (length `n`) into `re`/`im` (length `bins`).
    public func forward(_ x: UnsafePointer<Float>, re: UnsafeMutablePointer<Float>, im: UnsafeMutablePointer<Float>) {
        let half = n / 2
        // De-interleave even/odd samples: the zrop input layout.
        for k in 0..<half {
            inReal[k] = x[2 * k]
            inImag[k] = x[2 * k + 1]
        }
        vDSP_DFT_Execute(fwd, inReal, inImag, &outReal, &outImag)
        // zrop returns 2x the true DFT, with Nyquist packed into imag[0].
        re[0] = outReal[0] * 0.5
        im[0] = 0
        re[half] = outImag[0] * 0.5
        im[half] = 0
        for k in 1..<half {
            re[k] = outReal[k] * 0.5
            im[k] = outImag[k] * 0.5
        }
    }

    /// Inverse of `forward`: `bins` complex values back to `n` real samples.
    public func inverse(re: UnsafePointer<Float>, im: UnsafePointer<Float>, into y: UnsafeMutablePointer<Float>) {
        let half = n / 2
        inReal[0] = re[0]
        inImag[0] = re[half]
        for k in 1..<half {
            inReal[k] = re[k]
            inImag[k] = im[k]
        }
        vDSP_DFT_Execute(inv, inReal, inImag, &outReal, &outImag)
        // The zrop round trip scales by 2n relative to the forward output
        // above, of which we already removed a factor of 2.
        let scale = 1.0 / Float(n)
        for k in 0..<half {
            y[2 * k] = outReal[k] * scale
            y[2 * k + 1] = outImag[k] * scale
        }
    }

    /// Convenience allocating form, for non-real-time callers.
    public func forward(_ x: [Float]) -> (re: [Float], im: [Float]) {
        var re = [Float](repeating: 0, count: bins)
        var im = [Float](repeating: 0, count: bins)
        x.withUnsafeBufferPointer { xp in
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    forward(xp.baseAddress!, re: rp.baseAddress!, im: ip.baseAddress!)
                }
            }
        }
        return (re, im)
    }

    public func inverse(re: [Float], im: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: n)
        re.withUnsafeBufferPointer { rp in
            im.withUnsafeBufferPointer { ip in
                y.withUnsafeMutableBufferPointer { yp in
                    inverse(re: rp.baseAddress!, im: ip.baseAddress!, into: yp.baseAddress!)
                }
            }
        }
        return y
    }

    /// Smallest length >= `minimum` that vDSP's real DFT supports.
    public static func supportedLength(atLeast minimum: Int) -> Int {
        var best = Int.max
        for f in [1, 3, 5, 15] {
            var len = f * 16
            while len < minimum { len *= 2 }
            best = min(best, len)
        }
        return best
    }
}
