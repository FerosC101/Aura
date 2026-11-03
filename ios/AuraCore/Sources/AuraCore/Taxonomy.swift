import Foundation

/// The AURA sound taxonomy. Mirrors src/aura/classification/taxonomy.py.
///
/// Each case carries everything downstream components need -- which stream it
/// lives in, where it sits in the spectrum, whether the SafetyGuard protects it
/// -- plus the plain-language labels the app shows. The engine never uses the
/// display labels; the UI never uses the acoustic metadata.
public enum SoundCategory: Int, CaseIterable, Codable, Sendable, Identifiable {
    case humanSpeech, crowdSpeech, vehicleHorn, emergencySiren, engineNoise,
         generalTraffic, alarm, music, mechanicalNoise, announcement,
         bicycleBell, approachingVehicle, warningSignal, ambient

    public var id: Int { rawValue }

    public struct Info: Sendable {
        public let key: String
        public let label: String
        public let stream: StreamKind
        public let safetyCritical: Bool
        public let bandHz: (Float, Float)
        public let symbol: String
    }

    public var info: Info {
        switch self {
        case .humanSpeech: Info(key: "human_speech", label: "Nearby voices", stream: .primarySpeech, safetyCritical: false, bandHz: (85, 4000), symbol: "person.wave.2")
        case .crowdSpeech: Info(key: "crowd_speech", label: "Crowd noise", stream: .secondarySpeech, safetyCritical: false, bandHz: (150, 3500), symbol: "person.3")
        case .vehicleHorn: Info(key: "vehicle_horn", label: "Horn", stream: .transient, safetyCritical: true, bandHz: (300, 4000), symbol: "horn")
        case .emergencySiren: Info(key: "emergency_siren", label: "Sirens", stream: .transient, safetyCritical: true, bandHz: (500, 3000), symbol: "light.beacon.max")
        case .engineNoise: Info(key: "engine_noise", label: "Engine rumble", stream: .environment, safetyCritical: false, bandHz: (20, 300), symbol: "bus")
        case .generalTraffic: Info(key: "general_traffic", label: "Traffic", stream: .environment, safetyCritical: false, bandHz: (100, 6000), symbol: "car")
        case .alarm: Info(key: "alarm", label: "Alarms", stream: .transient, safetyCritical: true, bandHz: (800, 6000), symbol: "alarm")
        case .music: Info(key: "music", label: "Music", stream: .environment, safetyCritical: false, bandHz: (40, 12000), symbol: "music.note")
        case .mechanicalNoise: Info(key: "mechanical_noise", label: "Construction", stream: .environment, safetyCritical: false, bandHz: (60, 8000), symbol: "hammer")
        case .announcement: Info(key: "announcement", label: "Announcements", stream: .primarySpeech, safetyCritical: false, bandHz: (200, 4000), symbol: "speaker.wave.2")
        case .bicycleBell: Info(key: "bicycle_bell", label: "Bike bells", stream: .transient, safetyCritical: true, bandHz: (1500, 8000), symbol: "bicycle")
        case .approachingVehicle: Info(key: "approaching_vehicle", label: "Approaching vehicles", stream: .environment, safetyCritical: true, bandHz: (50, 4000), symbol: "car.side.arrowtriangle.up")
        case .warningSignal: Info(key: "warning_signal", label: "Warning signals", stream: .transient, safetyCritical: true, bandHz: (700, 5000), symbol: "exclamationmark.triangle")
        case .ambient: Info(key: "ambient", label: "Background hum", stream: .environment, safetyCritical: false, bandHz: (20, 20000), symbol: "wind")
        }
    }

    /// Context-specific wording. The same babble is "café chatter" when you
    /// are studying and "crowd noise" on a bus; the category underneath is one.
    public func label(in mode: ListeningMode, personLock: String? = nil) -> String {
        switch (self, mode) {
        case (.crowdSpeech, .care): return "Waiting room"
        case (.crowdSpeech, .focus): return "Café chatter"
        case (.mechanicalNoise, .care): return "Air-conditioning"
        case (.music, .care): return "Television"
        case (.humanSpeech, .care): return "The person speaking"
        case (.crowdSpeech, .social): return "Other conversations"
        case (.humanSpeech, _) where personLock != nil: return "\(personLock!)'s voice"
        case (.humanSpeech, .social): return "Friend's voice"
        case (.humanSpeech, .commute): return "Driver & conductor"
        default: return info.label
        }
    }

    /// Context-specific glyph, for the same reason as `label(in:)`. An
    /// air-conditioner drawn as a hammer is the kind of detail that tells a
    /// reader the screen was built for somewhere else.
    public func symbol(in mode: ListeningMode) -> String {
        switch (self, mode) {
        case (.mechanicalNoise, .care): return "wind"
        case (.music, .care): return "tv"
        default: return info.symbol
        }
    }

    public static var safetyCritical: [SoundCategory] { allCases.filter { $0.info.safetyCritical } }
}
