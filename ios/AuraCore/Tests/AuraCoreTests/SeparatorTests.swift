import XCTest
@testable import AuraCore

final class SeparatorTests: XCTestCase {
    /// Streams must sum back to the delayed input. The mixer's whole model
    /// depends on this: without it, bypass would not be transparent.
    func testStreamsSumToInput() {
        let x = Fixture.scene("psc_demo")
        let sep = SourceSeparator(sampleRate: 48000, hop: 960)
        var err: Double = 0, sig: Double = 0
        x.withUnsafeBufferPointer { p in
            var i = 0
            while i + 960 <= min(x.count, 960 * 300) {
                sep.separate(p.baseAddress! + i)
                if i >= 960 {
                    for j in 0..<960 {
                        let sum = sep.outputs.reduce(Float(0)) { $0 + $1[j] }
                        let ref = x[i - 960 + j]
                        err += Double((sum - ref) * (sum - ref)); sig += Double(ref * ref)
                    }
                }
                i += 960
            }
        }
        let snr = 10 * log10(sig / max(err, 1e-30))
        print(String(format: "reconstruction SNR %.1f dB", snr))
        XCTAssertGreaterThan(snr, 90)
    }

    private func route(_ file: String) -> [Double] {
        let x = Fixture.load("sources/\(file).wav")
        let sep = SourceSeparator(sampleRate: 48000, hop: 960)
        var e = [Double](repeating: 0, count: 4)
        x.withUnsafeBufferPointer { p in
            var i = 0
            while i + 960 <= x.count {
                sep.separate(p.baseAddress! + i)
                for k in 0..<4 { for v in sep.outputs[k] { e[k] += Double(v * v) } }
                i += 960
            }
        }
        let t = e.reduce(0, +) + 1e-18
        return e.map { $0 / t }
    }

    func testRouting() {
        let engine = route("engine_noise"), siren = route("emergency_siren")
        let speech = route("human_speech"), traffic = route("general_traffic")
        print("engine env \(engine[3]), siren transient \(siren[2]), speech primary \(speech[0]), traffic env \(traffic[3])")
        XCTAssertGreaterThan(engine[StreamKind.environment.rawValue], 0.9)
        XCTAssertGreaterThan(siren[StreamKind.transient.rawValue], 0.6)
        XCTAssertGreaterThan(speech[0] + speech[1], 0.5)
        XCTAssertLessThan(speech[StreamKind.transient.rawValue], 0.15)
        XCTAssertGreaterThan(traffic[StreamKind.environment.rawValue], 0.9, "road noise leaking into speech")
    }
}
