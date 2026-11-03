import AuraCore
import Foundation

/// Everything AURA writes to disk. There is no field that can hold audio:
/// voiceprints are embeddings, name templates are cepstral sequences.
struct StoredProfile: Codable {
    var userName = ""
    var onboarded = false
    var mode: ListeningMode = .commute
    var primaryUses: [String] = []
    /// mode -> category key -> gain in dB. The user's own slider positions.
    var overrides: [String: [String: Float]] = [:]
    var suppression: Float = 1
    var safetyEnabled: [String] = SoundCategory.safetyCritical.map(\.info.key)
    var learner = PreferenceLearner()
    var speakers: [SpeakerProfile] = []
    var selectedSpeaker: String?
    var nameAwareness = false
    var nameTemplates: [String: [[[Float]]]] = [:]
    var sessionInsights = true

    func override(_ m: ListeningMode, _ c: SoundCategory) -> Float? { overrides[m.rawValue]?[c.info.key] }

    func overrides(for m: ListeningMode) -> [SoundCategory: Float] {
        var out: [SoundCategory: Float] = [:]
        for c in SoundCategory.allCases { if let v = override(m, c) { out[c] = v } }
        return out
    }

    var safetySet: Set<SoundCategory> {
        Set(SoundCategory.safetyCritical.filter { safetyEnabled.contains($0.info.key) })
    }
}

/// Local-only persistence. The file is written atomically, protected with
/// complete file protection, and excluded from iCloud backup -- voiceprints do
/// not leave this device.
enum ProfileStore {
    static var url: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AURA", isDirectory: true)
        return dir.appendingPathComponent("profile.json")
    }

    static func load() -> StoredProfile {
        guard let data = try? Data(contentsOf: url),
              let p = try? JSONDecoder().decode(StoredProfile.self, from: data) else { return StoredProfile() }
        return p
    }

    static func save(_ p: StoredProfile) {
        do {
            var dir = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try dir.setResourceValues(values)
            let data = try JSONEncoder().encode(p)
            try data.write(to: url, options: [.atomic, .completeFileProtection])
        } catch {
            // Losing a preference write must never interrupt listening.
        }
    }

    static func delete() { try? FileManager.default.removeItem(at: url) }
}
