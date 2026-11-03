import AuraCore
import AVFoundation

/// The scripted scenes bundled with the app (rendered by the Python reference
/// engine, see scripts/export_ios_fixtures.py). A live demo cannot rely on the
/// room containing an ambulance.
struct DemoScene: Decodable, Identifiable, Hashable {
    struct Layer: Decodable, Hashable {
        let category: String
        let start: Double
        let end: Double
    }

    let key: String
    let title: String
    let subtitle: String
    let description: String
    let duration: Double
    let mode: String
    let expect: [String: String]
    let highlights: [String]
    let layers: [Layer]

    var id: String { key }
    var listeningMode: ListeningMode { ListeningMode(rawValue: mode) ?? .commute }
}

enum DemoCatalog {
    static let scenes: [DemoScene] = {
        guard let url = Bundle.main.url(forResource: "scenes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let s = try? JSONDecoder().decode([DemoScene].self, from: data) else { return [] }
        return s
    }()

    static func samples(_ key: String) -> [Float]? {
        guard let url = Bundle.main.url(forResource: key, withExtension: "wav"),
              let file = try? AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false),
              let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: buf)) != nil,
              let ch = buf.floatChannelData else { return nil }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(buf.frameLength)))
    }
}

extension SoundCategory {
    static func from(key: String) -> SoundCategory? { allCases.first { $0.info.key == key } }
}
