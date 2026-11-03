import Foundation

// Value types shared across the engine. Mirrors src/aura/types.py.

public enum ListeningMode: String, CaseIterable, Codable, Sendable, Identifiable {
    case care, focus, commute, social, walk, custom
    public var id: String { rawValue }
}

/// Ordered loudest to quietest, so `max(by: rank)` picks the most audible.
public enum SoundAction: String, Codable, Sendable {
    case enhance, preserve, reduce, suppress

    public var rank: Int {
        switch self {
        case .enhance: 3
        case .preserve: 2
        case .reduce: 1
        case .suppress: 0
        }
    }

    public static func forGain(_ db: Float) -> SoundAction {
        if db >= 2.0 { return .enhance }
        if db > -3.0 { return .preserve }
        if db > -12.0 { return .reduce }
        return .suppress
    }
}

public enum Priority: Int, Codable, Sendable, Comparable {
    case low = 0, normal = 1, high = 2, critical = 3
    public static func < (a: Priority, b: Priority) -> Bool { a.rawValue < b.rawValue }
}

/// The separated streams the mixer can address independently.
public enum StreamKind: Int, CaseIterable, Sendable {
    case primarySpeech = 0, secondarySpeech, transient, environment
}

public struct SoundEvent: Sendable {
    public var category: SoundCategory
    public var confidence: Float
    public var stream: StreamKind
    public var levelDB: Float

    public init(_ category: SoundCategory, confidence: Float, stream: StreamKind? = nil, levelDB: Float = -60) {
        self.category = category
        self.confidence = min(max(confidence, 0), 1)
        self.stream = stream ?? category.info.stream
        self.levelDB = levelDB
    }
}

public struct GainDecision: Sendable {
    public var category: SoundCategory
    public var action: SoundAction
    public var gainDB: Float
    public var priority: Priority
    public var alwaysAudible: Bool = false
    public var safetyOverride: Bool = false
    public var reason: String = ""
    public var stream: StreamKind
    public var confidence: Float = 1

    public var linearGain: Float { powf(10, gainDB / 20) }
}

/// Per-stage timing for one block, in milliseconds.
public struct LatencyBreakdown: Sendable {
    public var separationMS: Float = 0
    public var classificationMS: Float = 0
    public var policyMS: Float = 0
    public var mixingMS: Float = 0
    public var processingMS: Float { separationMS + classificationMS + policyMS + mixingMS }
}
