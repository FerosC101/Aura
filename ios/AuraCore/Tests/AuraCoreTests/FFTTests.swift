import XCTest
@testable import AuraCore

final class FFTTests: XCTestCase {
    /// The FFT must agree with a textbook DFT, or every feature ported from
    /// NumPy silently means something different.
    func testMatchesNaiveDFT() {
        for n in [64, 960, 1920] {
            let fft = RealFFT(n)
            var rng = SystemRandomNumberGenerator()
            let x = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
            let (re, im) = fft.forward(x)
            for k in [0, 1, 7, n / 4, n / 2] {
                var sr = 0.0, si = 0.0
                for t in 0..<n {
                    let a = -2.0 * Double.pi * Double(k * t) / Double(n)
                    sr += Double(x[t]) * cos(a)
                    si += Double(x[t]) * sin(a)
                }
                XCTAssertEqual(Double(re[k]), sr, accuracy: 1e-2 * Double(n).squareRoot(), "n=\(n) k=\(k) re")
                XCTAssertEqual(Double(im[k]), si, accuracy: 1e-2 * Double(n).squareRoot(), "n=\(n) k=\(k) im")
            }
        }
    }

    func testRoundTrip() {
        let fft = RealFFT(960)
        let x = (0..<960).map { Float(sin(Double($0) * 0.37)) + 0.3 }
        let (re, im) = fft.forward(x)
        let y = fft.inverse(re: re, im: im)
        for i in 0..<960 { XCTAssertEqual(x[i], y[i], accuracy: 1e-4) }
    }

    func testSupportedLengths() {
        XCTAssertEqual(RealFFT.supportedLength(atLeast: 1920), 1920)
        XCTAssertEqual(RealFFT.supportedLength(atLeast: 1921), 2048)
        XCTAssertEqual(RealFFT.supportedLength(atLeast: 400), 480)
    }
}
