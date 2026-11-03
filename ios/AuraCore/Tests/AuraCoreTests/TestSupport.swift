import Foundation
import XCTest
@testable import AuraCore

/// Fixtures are rendered by scripts/export_ios_fixtures.py from the Python
/// reference engine, so the Swift port is tested on exactly the audio the
/// Python engine is benchmarked on.
enum Fixture {
    static let sampleRate: Float = 48000
    static let hop = 960

    static func url(_ path: String) -> URL {
        Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appendingPathComponent(path)
    }

    /// Demo scenes live in the app bundle; read them from the source tree.
    static func sceneURL(_ key: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("AuraApp/Resources/Scenes/\(key).wav")
    }

    static func load(_ path: String) -> [Float] { readWAV(url(path)) }
    static func scene(_ key: String) -> [Float] { readWAV(sceneURL(key)) }

    /// Minimal 16-bit PCM mono WAV reader.
    static func readWAV(_ url: URL) -> [Float] {
        guard let data = try? Data(contentsOf: url) else {
            XCTFail("missing fixture \(url.path)")
            return []
        }
        var offset = 12
        while offset + 8 <= data.count {
            let id = String(data: data[offset..<offset + 4], encoding: .ascii)
            let size = Int(data[offset + 4..<offset + 8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
            if id == "data" {
                let body = data[(offset + 8)..<min(offset + 8 + size, data.count)]
                return body.withUnsafeBytes { raw in
                    raw.bindMemory(to: Int16.self).map { Float($0) / 32767 }
                }
            }
            offset += 8 + size
        }
        return []
    }

    /// Run the classifier over a signal; returns per-category frame counts.
    static func classify(_ x: [Float], warmup: Int = 25) -> (tallies: [SoundCategory: Int], scored: Int) {
        let clf = SoundClassifier(sampleRate: sampleRate, frameLength: hop)
        var tallies: [SoundCategory: Int] = [:]
        var scored = 0
        var i = 0, frame = 0
        x.withUnsafeBufferPointer { p in
            while i + hop <= x.count {
                let events = clf.classify(mix: p.baseAddress! + i, count: hop)
                if frame >= warmup {
                    scored += 1
                    for e in events { tallies[e.category, default: 0] += 1 }
                }
                i += hop; frame += 1
            }
        }
        return (tallies, max(scored, 1))
    }
}

func bandEnergyDB(_ x: ArraySlice<Float>, _ sr: Float, _ lo: Float, _ hi: Float) -> Float {
    let n = RealFFT.supportedLength(atLeast: x.count)
    let fft = RealFFT(n)
    var buf = [Float](repeating: 0, count: n)
    let w = hanning(x.count)
    for (i, v) in x.enumerated() { buf[i] = v * w[i] }
    let (re, im) = fft.forward(buf)
    var s: Double = 0
    for k in 0..<re.count {
        let f = Float(k) * sr / Float(n)
        if f >= lo && f < hi { s += Double(re[k] * re[k] + im[k] * im[k]) }
    }
    return Float(10 * log10(s + 1e-18))
}
