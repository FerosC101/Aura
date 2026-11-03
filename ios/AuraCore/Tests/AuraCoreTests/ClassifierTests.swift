import XCTest
@testable import AuraCore

final class ClassifierTests: XCTestCase {
    /// Same regression floors as tests/test_classification.py in the Python
    /// reference. If the port drifts, this is where it shows.
    static let expected: [(String, SoundCategory, Double)] = [
        ("human_speech", .humanSpeech, 0.50),
        ("crowd_speech", .crowdSpeech, 0.70),
        ("engine_noise", .engineNoise, 0.80),
        ("general_traffic", .generalTraffic, 0.70),
        ("vehicle_horn", .vehicleHorn, 0.70),
        ("emergency_siren", .emergencySiren, 0.70),
        ("alarm", .alarm, 0.50),
        ("music", .music, 0.70),
        ("mechanical_noise", .mechanicalNoise, 0.70),
        ("announcement", .announcement, 0.50),
        ("approaching_vehicle", .approachingVehicle, 0.70),
        ("warning_signal", .warningSignal, 0.50),
    ]

    func testIsolatedSourcesAreDetected() {
        var report: [String] = []
        var failures = 0
        for (file, cat, floor) in Self.expected {
            let (t, n) = Fixture.classify(Fixture.load("sources/\(file).wav"))
            let recall = Double(t[cat, default: 0]) / Double(n)
            let top = t.sorted { $0.value > $1.value }.prefix(3).map { "\($0.key.info.key):\($0.value * 100 / n)%" }
            report.append(String(format: "%-20@ %5.0f%%  %@", file as NSString, recall * 100, top.joined(separator: ", ")))
            if recall < floor { failures += 1 }
        }
        print("\n" + report.joined(separator: "\n"))
        XCTAssertEqual(failures, 0, "sources below their recall floor:\n" + report.joined(separator: "\n"))
    }

    func testSirenAndHornAreDistinguished() {
        for (truth, rival) in [(SoundCategory.emergencySiren, SoundCategory.vehicleHorn), (.vehicleHorn, .emergencySiren)] {
            let (t, _) = Fixture.classify(Fixture.load("sources/\(truth.info.key).wav"))
            XCTAssertGreaterThan(t[truth, default: 0], t[rival, default: 0], "\(truth) not preferred over \(rival)")
        }
    }

    func testSilenceIsAmbientOnly() {
        let (t, _) = Fixture.classify([Float](repeating: 0, count: 960 * 60))
        XCTAssertEqual(Set(t.keys), [.ambient])
    }

    func testFitsTheRealtimeBudget() {
        let x = Fixture.load("sources/general_traffic.wav")
        let clf = SoundClassifier(sampleRate: 48000, frameLength: 960)
        let frames = x.count / 960
        let start = DispatchTime.now().uptimeNanoseconds
        x.withUnsafeBufferPointer { p in
            for f in 0..<frames { _ = clf.classify(mix: p.baseAddress! + f * 960, count: 960) }
        }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(frames)
        print(String(format: "classifier: %.3f ms/frame", ms))
        XCTAssertLessThan(ms, 10, "classification alone would eat half the 20 ms budget")
    }
}
