import XCTest
@testable import AuraCore

final class PolicyTests: XCTestCase {
    func ctx(_ m: ListeningMode) -> PolicyContext { var c = PolicyContext(); c.mode = m; return c }

    /// The worked examples from the brief.
    func testBriefExamples() {
        let p = PolicyEngine()
        let horn = p.explain(.vehicleHorn, ctx(.walk))
        XCTAssertEqual(horn.priority, .critical)
        XCTAssertGreaterThanOrEqual(horn.gainDB, 0)
        let engine = p.explain(.engineNoise, ctx(.commute))
        XCTAssertEqual(engine.action, .suppress)
        XCTAssertLessThanOrEqual(engine.gainDB, -12)
        XCTAssertEqual(p.explain(.humanSpeech, ctx(.social)).action, .enhance)
    }

    /// Every safety category, every mode, with a hostile override: nothing
    /// can take a siren below the floor.
    func testSafetySoundsAreNeverSilenced() {
        let p = PolicyEngine()
        for c in SoundCategory.safetyCritical {
            for m in ListeningMode.allCases {
                var x = ctx(m)
                x.userOverrides[c] = -40
                x.suppressionStrength = 1.5
                x.learnedBias[c] = -6
                let d = p.explain(c, x)
                XCTAssertGreaterThanOrEqual(d.gainDB, SafetyGuard.minGainDB, "\(c) in \(m)")
                XCTAssertTrue(d.alwaysAudible)
            }
        }
    }

    func testUncertainSafetySoundIsKept() {
        let (d, overridden) = SafetyGuard().apply(
            GainDecision(category: .emergencySiren, action: .suppress, gainDB: -18, priority: .low, stream: .transient),
            SoundEvent(.emergencySiren, confidence: 0.45, levelDB: -30), sceneLevelDB: -30)
        XCTAssertTrue(overridden)
        XCTAssertGreaterThanOrEqual(d.gainDB, 0)
    }

    func testPersonLockPromotesAndDucks() {
        let p = PolicyEngine()
        var locked = ctx(.social); locked.personLockActive = true; locked.personLockName = "Vince"
        XCTAssertGreaterThan(p.explain(.humanSpeech, locked).gainDB, p.explain(.humanSpeech, ctx(.social)).gainDB)
        XCTAssertLessThan(p.explain(.crowdSpeech, locked).gainDB, p.explain(.crowdSpeech, ctx(.social)).gainDB)
    }

    func testLearningIsBoundedAndSafe() {
        var l = PreferenceLearner()
        for _ in 0..<50 { l.observe(.commute, .announcement, correctionDB: 20) }
        XCTAssertLessThanOrEqual(l.bias(.commute, .announcement), PreferenceLearner.maxBiasDB)
        XCTAssertGreaterThan(l.bias(.commute, .announcement), 3)
        XCTAssertFalse(l.observe(.walk, .emergencySiren, correctionDB: -8))
        XCTAssertEqual(l.bias(.walk, .emergencySiren), 0)
    }
}

final class MixerTests: XCTestCase {
    func testShaperHitsTargets() {
        let s = BandShaper(sampleRate: 48000)
        let target: [Float] = [-18, -18, -6, 0, 0, 0]
        s.setGains(target)
        for (i, f) in BandShaper.bands.enumerated() {
            XCTAssertEqual(s.response(at: f), target[i], accuracy: 1.5, "band \(f) Hz")
        }
    }

    func testEngineSuppressionLeavesHornBandAlone() {
        let m = AudioMixer(sampleRate: 48000, maxBlock: 960)
        let streams = (0..<4).map { _ in (0..<960).map { _ in Float.random(in: -0.1...0.1) } }
        var out = [Float](repeating: 0, count: 960)
        let d = GainDecision(category: .engineNoise, action: .suppress, gainDB: -18, priority: .low, stream: .environment, confidence: 0.95)
        for _ in 0..<80 { out.withUnsafeMutableBufferPointer { m.mix(streams, [d], bypass: false, into: $0.baseAddress!, count: 960) } }
        XCTAssertLessThan(m.bandDB[0], -10)
        XCTAssertLessThan(m.bandDB[1], -10)
        for b in 3..<6 { XCTAssertLessThan(abs(m.bandDB[b]), 2, "horn band \(b) disturbed") }
    }

    func testLimiterPreventsClipping() {
        let l = SoftLimiter(sampleRate: 48000)
        var x = [Float](repeating: 4, count: 960)
        x.withUnsafeMutableBufferPointer { l.process($0.baseAddress!, count: 960) }
        XCTAssertLessThanOrEqual(x.map(abs).max()!, powf(10, -1.0 / 20) + 1e-3)
    }
}

final class HopTests: XCTestCase {
    func testHopLengthsAreTransformable() {
        XCTAssertEqual(AuraProcessor.hopLength(for: 48000), 960)
        for sr: Float in [44100, 48000, 16000, 24000] {
            let h = AuraProcessor.hopLength(for: sr)
            // Both the hop and the WOLA frame must be vDSP-friendly lengths.
            _ = RealFFT(h); _ = RealFFT(2 * h)
            XCTAssertLessThan(abs(Float(h) / sr - 0.02), 0.006, "\(sr) Hz -> \(h)")
        }
    }
}
