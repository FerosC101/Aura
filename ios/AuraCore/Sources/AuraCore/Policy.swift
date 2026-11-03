import Foundation

// MARK: - Modes (port of policy/modes.py)

public struct ModeProfile: Sendable {
    public let mode: ListeningMode
    public let title: String
    public let tagline: String
    public let intents: [SoundCategory: SoundAction]
    public let criticalCategories: Set<SoundCategory>
    public let defaultAction: SoundAction
    /// Walk mode suppresses at 0.7x: over-suppression while walking is a
    /// safety problem, not a matter of taste.
    public let suppressionScale: Float

    public func intent(_ c: SoundCategory) -> SoundAction { intents[c] ?? defaultAction }

    public func priority(_ c: SoundCategory) -> Priority {
        if criticalCategories.contains(c) { return .critical }
        if mode == .commute && c == .announcement { return .high }
        switch intent(c) {
        case .enhance: return .high
        case .preserve: return .normal
        default: return .low
        }
    }

    public static let actionGain: [SoundAction: Float] = [.enhance: 5, .preserve: 0, .reduce: -9, .suppress: -16]

    public static func profile(_ mode: ListeningMode) -> ModeProfile {
        switch mode {
        case .care:
            // Port of CARE in policy/modes.py. The difference from .social is
            // crowdSpeech and music: in a clinic the competing sound is other
            // speech and a television, and that is what buries the words in
            // front of you. Safety stays preserved regardless.
            return ModeProfile(mode: .care, title: "Care", tagline: "Clearer conversations when they matter most.", intents: [
                .humanSpeech: .enhance, .announcement: .preserve, .crowdSpeech: .suppress,
                // Reduced, not suppressed -- see the note on CARE in modes.py.
                .music: .reduce, .mechanicalNoise: .reduce, .ambient: .reduce,
                .generalTraffic: .reduce, .engineNoise: .reduce,
                .alarm: .preserve, .emergencySiren: .preserve, .vehicleHorn: .preserve,
                .warningSignal: .preserve, .bicycleBell: .preserve, .approachingVehicle: .preserve,
            ], criticalCategories: [], defaultAction: .reduce, suppressionScale: 1.2)
        case .focus:
            return ModeProfile(mode: .focus, title: "Focus", tagline: "Find your focus today.", intents: [
                .humanSpeech: .preserve, .crowdSpeech: .suppress, .announcement: .preserve,
                .generalTraffic: .suppress, .engineNoise: .suppress, .mechanicalNoise: .suppress,
                .music: .reduce, .ambient: .suppress, .alarm: .preserve, .emergencySiren: .preserve,
                .vehicleHorn: .preserve, .warningSignal: .preserve, .bicycleBell: .reduce, .approachingVehicle: .reduce,
            ], criticalCategories: [], defaultAction: .reduce, suppressionScale: 1.15)
        case .commute:
            return ModeProfile(mode: .commute, title: "Commute", tagline: "Stay aware. Stay sane.", intents: [
                .humanSpeech: .preserve, .announcement: .enhance, .crowdSpeech: .reduce,
                .vehicleHorn: .preserve, .emergencySiren: .preserve, .warningSignal: .preserve, .alarm: .preserve,
                .engineNoise: .suppress, .generalTraffic: .reduce, .mechanicalNoise: .reduce, .music: .reduce,
                .ambient: .reduce, .approachingVehicle: .preserve, .bicycleBell: .preserve,
            ], criticalCategories: [], defaultAction: .reduce, suppressionScale: 1.0)
        case .walk:
            return ModeProfile(mode: .walk, title: "Walk", tagline: "More of the world. Less of the noise.", intents: [
                .approachingVehicle: .enhance, .vehicleHorn: .enhance, .emergencySiren: .enhance,
                .bicycleBell: .enhance, .warningSignal: .enhance, .alarm: .preserve, .humanSpeech: .preserve,
                .announcement: .preserve, .crowdSpeech: .reduce, .generalTraffic: .reduce, .engineNoise: .reduce,
                .mechanicalNoise: .reduce, .music: .reduce, .ambient: .reduce,
            ], criticalCategories: [.approachingVehicle, .vehicleHorn, .emergencySiren, .bicycleBell, .warningSignal],
            defaultAction: .preserve, suppressionScale: 0.7)
        case .social:
            return ModeProfile(mode: .social, title: "Social", tagline: "Conversations that count.", intents: [
                .humanSpeech: .enhance, .crowdSpeech: .suppress, .announcement: .preserve, .music: .reduce,
                .generalTraffic: .suppress, .engineNoise: .suppress, .mechanicalNoise: .suppress, .ambient: .suppress,
                .alarm: .preserve, .emergencySiren: .preserve, .vehicleHorn: .preserve, .warningSignal: .preserve,
                .bicycleBell: .preserve, .approachingVehicle: .preserve,
            ], criticalCategories: [], defaultAction: .reduce, suppressionScale: 1.0)
        case .custom:
            return ModeProfile(mode: .custom, title: "Custom", tagline: "Your own mix.", intents: [:],
                               criticalCategories: [], defaultAction: .preserve, suppressionScale: 1.0)
        }
    }
}

// MARK: - Safety (port of policy/safety.py)

/// The last layer to touch a decision, and the only one nothing can override.
public struct SafetyGuard: Sendable {
    public static let minGainDB: Float = -3
    public static let uncertainLow: Float = 0.30
    public static let uncertainHigh: Float = 0.72
    public static let maxLiftDB: Float = 8

    /// Safety categories the user kept on. Defaults to all of them.
    public var enabled: Set<SoundCategory> = Set(SoundCategory.safetyCritical)

    public init() {}

    public func isProtected(_ c: SoundCategory) -> Bool { c.info.safetyCritical && enabled.contains(c) }

    public func apply(_ d: GainDecision, _ e: SoundEvent, sceneLevelDB: Float) -> (GainDecision, Bool) {
        guard isProtected(d.category) else { return (d, false) }
        var out = d
        out.alwaysAudible = true
        out.priority = .critical
        var overridden = false
        var reasons: [String] = []
        // 1. Hard floor.
        if out.gainDB < Self.minGainDB {
            out.gainDB = Self.minGainDB; overridden = true; reasons.append("safety floor")
        }
        // 2. Lift a quiet warning above a loud scene.
        let deficit = sceneLevelDB - e.levelDB
        if deficit > 6 {
            let lift = min(Self.maxLiftDB, deficit - 6)
            if lift > 0.5 { out.gainDB = max(out.gainDB, lift); overridden = true; reasons.append("lifted over scene") }
        }
        // 3. Doubt resolves towards audibility.
        if e.confidence >= Self.uncertainLow && e.confidence < Self.uncertainHigh && out.gainDB < 0 {
            out.gainDB = 0; overridden = true; reasons.append("uncertain, kept audible")
        }
        if overridden {
            out.safetyOverride = true
            out.action = SoundAction.forGain(out.gainDB)
            out.reason = reasons.joined(separator: " · ")
        }
        return (out, overridden)
    }
}

// MARK: - Policy engine (port of policy/engine.py)

public struct PolicyContext: Sendable {
    public var mode: ListeningMode = .commute
    public var enabled = true
    public var personLockActive = false
    public var personLockName: String?
    public var nameDetected = false
    public var userOverrides: [SoundCategory: Float] = [:]
    public var learnedBias: [SoundCategory: Float] = [:]
    public var suppressionStrength: Float = 1
    public var sceneLevelDB: Float = -30
    public init() {}
}

/// Layers, each overridable only by the one beneath: mode -> user override ->
/// learned bias -> Person Lock -> Name Awareness -> confidence -> SafetyGuard.
public final class PolicyEngine {
    public static let personLockBoostDB: Float = 6
    public static let personLockDuckDB: Float = -7
    public static let nameBoostDB: Float = 5

    public var safety = SafetyGuard()
    public private(set) var lastSafetyOverride = false

    public init() {}

    public func decide(_ events: [SoundEvent], _ ctx: PolicyContext) -> [GainDecision] {
        guard ctx.enabled else {
            lastSafetyOverride = false
            return events.map {
                GainDecision(category: $0.category, action: .preserve, gainDB: 0, priority: .normal,
                             reason: "AURA off", stream: $0.stream, confidence: $0.confidence)
            }
        }
        let profile = ModeProfile.profile(ctx.mode)
        var any = false
        var out: [GainDecision] = []
        out.reserveCapacity(events.count)
        for e in events {
            let (d, o) = safety.apply(decideOne(e, profile, ctx), e, sceneLevelDB: ctx.sceneLevelDB)
            any = any || o
            out.append(d)
        }
        lastSafetyOverride = any
        return out.sorted { ($0.priority, $0.confidence) > ($1.priority, $1.confidence) }
    }

    /// What would happen to `category` right now -- drives the mixer rows.
    public func explain(_ c: SoundCategory, _ ctx: PolicyContext) -> GainDecision {
        let e = SoundEvent(c, confidence: 0.9, levelDB: ctx.sceneLevelDB)
        guard ctx.enabled else {
            return GainDecision(category: c, action: .preserve, gainDB: 0, priority: .normal, reason: "AURA off", stream: e.stream)
        }
        return safety.apply(decideOne(e, ModeProfile.profile(ctx.mode), ctx), e, sceneLevelDB: ctx.sceneLevelDB).0
    }

    func decideOne(_ e: SoundEvent, _ p: ModeProfile, _ ctx: PolicyContext) -> GainDecision {
        let c = e.category
        var action = p.intent(c)
        var gain = ModeProfile.actionGain[action]!
        var priority = p.priority(c)
        var reason = "\(p.title): \(action.rawValue)"

        if gain < 0 { gain *= p.suppressionScale * ctx.suppressionStrength }

        if let o = ctx.userOverrides[c] {
            gain = o; action = .forGain(o); priority = max(priority, .normal); reason = "you set \(Int(o.rounded())) dB"
        }
        if let b = ctx.learnedBias[c], abs(b) > 0.25 {
            gain += b; reason += " · learned"
        }
        if ctx.personLockActive {
            if e.stream == .primarySpeech {
                gain += Self.personLockBoostDB; priority = .high; reason += " · \(ctx.personLockName ?? "Person Lock")"
            } else if e.stream == .secondarySpeech {
                gain += Self.personLockDuckDB * ctx.suppressionStrength; reason += " · ducked"
            }
        }
        if ctx.nameDetected, e.stream == .primarySpeech || e.stream == .secondarySpeech {
            gain = max(gain, 0) + Self.nameBoostDB; priority = max(priority, .high); reason += " · your name"
        }
        // Tentative detections produce tentative actions.
        let w = confidenceWeight(e.confidence)
        if w < 0.999 { gain *= w }
        gain = clamp(gain, -40, 12)
        return GainDecision(category: c, action: .forGain(gain), gainDB: gain, priority: priority,
                            alwaysAudible: c.info.safetyCritical, reason: reason, stream: e.stream, confidence: e.confidence)
    }

    func confidenceWeight(_ c: Float) -> Float {
        c <= SafetyGuard.uncertainLow ? 0 : clamp((c - SafetyGuard.uncertainLow) / (0.85 - SafetyGuard.uncertainLow), 0, 1)
    }
}

// MARK: - Preference learning (port of profile/learning.py)

public struct PreferenceRecord: Codable, Sendable {
    public var biasDB: Float = 0
    public var weight: Float = 0
    public var observations = 0
    public var lastSeen = Date()
}

/// Learns per-(mode, category) biases from manual corrections. Bounded to
/// +/-6 dB, decays with a 21-day half-life, and never learns to quieten a
/// safety sound. Shaped as a bandit interface so a contextual bandit can
/// replace it.
public struct PreferenceLearner: Codable, Sendable {
    public static let maxBiasDB: Float = 6
    public var enabled = true
    public var records: [String: PreferenceRecord] = [:]

    public init() {}

    static func key(_ m: ListeningMode, _ c: SoundCategory) -> String { "\(m.rawValue)/\(c.info.key)" }

    @discardableResult
    public mutating func observe(_ m: ListeningMode, _ c: SoundCategory, correctionDB: Float, now: Date = Date()) -> Bool {
        guard enabled, abs(correctionDB) >= 1 else { return false }
        if c.info.safetyCritical && correctionDB < 0 { return false }
        var r = records[Self.key(m, c)] ?? PreferenceRecord()
        let target = clamp(correctionDB, -Self.maxBiasDB, Self.maxBiasDB)
        r.biasDB = clamp(r.biasDB + 0.28 * (target - r.biasDB), -Self.maxBiasDB, Self.maxBiasDB)
        r.observations += 1
        r.weight = min(1, Float(r.observations) / 4)
        r.lastSeen = now
        records[Self.key(m, c)] = r
        return true
    }

    public func bias(_ m: ListeningMode, _ c: SoundCategory, now: Date = Date()) -> Float {
        guard enabled, let r = records[Self.key(m, c)] else { return 0 }
        let days = max(0, now.timeIntervalSince(r.lastSeen) / 86400)
        return r.biasDB * r.weight * Float(pow(0.5, days / 21))
    }

    public func biases(_ m: ListeningMode) -> [SoundCategory: Float] {
        var out: [SoundCategory: Float] = [:]
        for c in SoundCategory.allCases { let b = bias(m, c); if abs(b) > 0.1 { out[c] = b } }
        return out
    }

    public mutating func forgetAll() -> Int { let n = records.count; records = [:]; return n }
}
