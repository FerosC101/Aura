import XCTest
@testable import AuraCore

/// Whole-engine tests on the bundled demo scenes: the claims the app makes,
/// measured on the output signal.
final class EngineTests: XCTestCase {
    func run(_ x: [Float], mode: ListeningMode, comparison: Comparison = .aura,
             tally: ((AuraProcessor) -> Void)? = nil) -> (out: [Float], proc: AuraProcessor) {
        let p = AuraProcessor()
        p.update { $0.context.mode = mode; $0.comparison = comparison }
        var out = [Float](repeating: 0, count: x.count)
        let hop = p.hop
        x.withUnsafeBufferPointer { xi in
            out.withUnsafeMutableBufferPointer { yo in
                var i = 0
                while i + hop <= x.count {
                    p.processHop(xi.baseAddress! + i, yo.baseAddress! + i)
                    tally?(p)
                    i += hop
                }
            }
        }
        return (out, p)
    }

    func testSelectiveTransformation() {
        let x = Fixture.scene("psc_demo")
        let (y, p) = run(x, mode: .commute)
        let engine = bandEnergyDB(y[...], 48000, 20, 300) - bandEnergyDB(x[...], 48000, 20, 300)
        let horn = bandEnergyDB(y[...], 48000, 400, 3500) - bandEnergyDB(x[...], 48000, 400, 3500)
        print(String(format: "engine band %+.1f dB, horn/siren band %+.1f dB, %.3f ms/hop", engine, horn, p.snapshot().processingMS))
        XCTAssertLessThan(engine, -6)
        XCTAssertGreaterThan(horn, -3)
        XCTAssertLessThan(engine, horn - 4, "suppression was not selective")
    }

    func testSirenSurvivesWhereANCBuriesIt() {
        let x = Fixture.scene("psc_demo")
        let aura = run(x, mode: .commute).out
        let anc = run(x, mode: .commute, comparison: .anc).out
        // Siren sounds 7.5-11 s in the PSC scene.
        let a = Int(8.0 * 48000), b = Int(10.5 * 48000)
        let base = bandEnergyDB(x[a..<b], 48000, 500, 3000)
        let auraDB = bandEnergyDB(aura[a..<b], 48000, 500, 3000) - base
        let ancDB = bandEnergyDB(anc[a..<b], 48000, 500, 3000) - base
        print(String(format: "siren: AURA %+.1f dB, ANC %+.1f dB", auraDB, ancDB))
        XCTAssertGreaterThan(auraDB, -3)
        XCTAssertGreaterThan(auraDB, ancDB + 4)
    }

    func testRawIsTransparent() {
        let x = Array(Fixture.scene("psc_demo").prefix(48000 * 3))
        let (y, p) = run(x, mode: .commute, comparison: .raw)
        let d = p.separator.algorithmicDelaySamples
        var err: Double = 0, sig: Double = 0
        for i in 0..<(x.count - d) { let e = Double(y[i + d] - x[i]); err += e * e; sig += Double(x[i] * x[i]) }
        XCTAssertGreaterThan(10 * log10(sig / max(err, 1e-30)), 40)
    }

    /// Each scene's scripted expectations, with the same two-miss allowance
    /// as the Python suite (acoustic masking is real).
    func testScenesBehaveAsScripted() throws {
        struct Scene: Decodable { let key: String; let mode: String; let expect: [String: String] }
        let manifest = Fixture.sceneURL("psc_demo").deletingLastPathComponent().appendingPathComponent("scenes.json")
        let scenes = try JSONDecoder().decode([Scene].self, from: Data(contentsOf: manifest))
        for s in scenes {
            var seen: [String: [String: Int]] = [:]
            var n = 0
            _ = run(Fixture.scene(s.key), mode: ListeningMode(rawValue: s.mode)!) { p in
                n += 1
                guard n % 4 == 0 else { return }
                for c in p.snapshot().sounds { seen[c.category.info.key, default: [:]][c.action.rawValue, default: 0] += 1 }
            }
            var matched = 0
            var notes: [String] = []
            for (cat, want) in s.expect {
                guard let acts = seen[cat], let modal = acts.max(by: { $0.value < $1.value })?.key else { notes.append("\(cat): missed"); continue }
                if modal == want { matched += 1 } else { notes.append("\(cat): \(modal) (want \(want))") }
            }
            print("\(s.key): \(matched)/\(s.expect.count) \(notes)")
            XCTAssertGreaterThanOrEqual(matched, s.expect.count - 2, "\(s.key): \(notes)")
        }
    }

    func testStreamingProducesContinuousOutput() {
        let p = AuraProcessor()
        let x = Array(Fixture.scene("cafe").prefix(48000 * 2))
        var y = [Float](repeating: 0, count: 256)
        var produced = 0
        x.withUnsafeBufferPointer { xi in
            var i = 0
            // Odd IO buffer sizes, as a real device delivers.
            while i + 256 <= x.count {
                p.pushInput(xi.baseAddress! + i, 256)
                y.withUnsafeMutableBufferPointer { p.pullOutput($0.baseAddress!, 256) }
                produced += 256; i += 256
            }
        }
        XCTAssertEqual(p.snapshot().dropouts, 0)
        XCTAssertGreaterThan(produced, 48000)
    }
}

final class SpeakerTests: XCTestCase {
    func testPersonLockAcceptsAndRejects() throws {
        let names = ["vince", "maria", "ben", "ana"]
        for target in names {
            _ = target
        }
        var ok = 0
        for speaking in names {
            let lock = PersonLock(sampleRate: 48000)
            for n in names {
                _ = try lock.enrol(name: n, clips: [Fixture.load("voices/\(n)_enrol0.wav"), Fixture.load("voices/\(n)_enrol1.wav")],
                                   sampleRate: 48000, consent: true)
            }
            lock.selectedID = lock.profiles.first { $0.name == "vince" }!.id
            let live = Fixture.load("voices/\(speaking)_live.wav")
            var s = LockState()
            var i = 0
            while i + 960 <= live.count { s = lock.update(Array(live[i..<i + 960]), speechPresent: true); i += 960 }
            let expected = speaking == "vince"
            print("\(speaking): active=\(s.active) sim=\(s.similarity) margin=\(s.margin)")
            if s.active == expected { ok += 1 }
        }
        XCTAssertEqual(ok, names.count)
    }

    func testEnrolmentNeedsConsent() {
        XCTAssertThrowsError(try PersonLock(sampleRate: 48000).enrol(
            name: "x", clips: [Fixture.load("voices/vince_enrol0.wav")], sampleRate: 48000, consent: false))
    }

    func testProfileHoldsNoAudio() throws {
        let lock = PersonLock(sampleRate: 48000)
        let p = try lock.enrol(name: "Vince", clips: [Fixture.load("voices/vince_enrol0.wav")], sampleRate: 48000, consent: true)
        XCTAssertLessThan(p.centroid.count, 100)
        let json = try JSONEncoder().encode(p)
        XCTAssertLessThan(json.count, 4000, "a profile holding audio would be orders of magnitude larger")
    }

    func testNameSpotting() {
        let s = NameSpotter(sampleRate: 48000)
        XCTAssertTrue(s.enrol("Vince", clips: [Fixture.load("words/vince_enrol0.wav"), Fixture.load("words/vince_enrol1.wav")], sampleRate: 48000))
        for w in ["vince_same", "vince_high", "vince_slow"] {
            XCTAssertEqual(s.detect(Fixture.load("words/\(w).wav"), sampleRate: 48000)?.0, "Vince", w)
        }
        for w in ["maria", "hello", "ben", "ana"] {
            XCTAssertNil(s.detect(Fixture.load("words/\(w).wav"), sampleRate: 48000), w)
        }
    }
}
