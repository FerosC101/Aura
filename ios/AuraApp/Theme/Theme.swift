import AuraCore
import SwiftUI

/// Brand tokens. Warm charcoal rather than cold slate, with colour reserved
/// for meaning: indigo for what AURA brings forward, aqua for what it lifts
/// above the rest, peach for what it steps back, coral for what it will never
/// touch. Nothing here is decorative -- every hue answers "what happened to
/// this sound?", which is the only question the interface has to answer.
enum Palette {
    static let bgTop = Color(hex: 0x1B1823)
    static let bgBottom = Color(hex: 0x0D0B12)
    static let card = Color(hex: 0x221F2D)
    static let cardHi = Color(hex: 0x2A2637)
    static let stroke = Color.white.opacity(0.07)

    static let text = Color(hex: 0xF5F3F8)
    static let text2 = Color(hex: 0xADA7BC)
    static let text3 = Color(hex: 0x746D85)

    static let sky = Color(hex: 0xA6A4F7)        // bars, "let through" -- lavender
    static let skyDeep = Color(hex: 0x7B78EE)    // plus glyphs -- electric indigo
    static let tileSelected = Color(hex: 0xF1EEF9)
    static let tileSelectedInk = Color(hex: 0x4A46C4)
    static let quiet = Color(hex: 0x6A6480)      // bars, "turned down"
    static let sunrise = Color(hex: 0xF4C49B)    // warm peach
    static let safety = Color(hex: 0xF2948C)     // soft coral -- never suppressed
    static let enhance = Color(hex: 0x6FD6CE)    // aqua -- brought forward
}

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xff) / 255, green: Double((hex >> 8) & 0xff) / 255,
                  blue: Double(hex & 0xff) / 255, opacity: opacity)
    }
}

enum Type_ {
    static func light(_ size: CGFloat) -> Font { .system(size: size, weight: .light) }
    static func regular(_ size: CGFloat) -> Font { .system(size: size, weight: .regular) }
    static func medium(_ size: CGFloat) -> Font { .system(size: size, weight: .medium) }
    static func semibold(_ size: CGFloat) -> Font { .system(size: size, weight: .semibold) }
    /// The handwritten accents ("A quieter, brighter tomorrow").
    static func script(_ size: CGFloat) -> Font { .custom("SnellRoundhand", size: size) }
}

/// The ΛURΛ wordmark: ultralight, widely tracked, with the chevron A.
struct Wordmark: View {
    var size: CGFloat = 22
    var glow = false

    var body: some View {
        Text("ΛURΛ")
            .font(.system(size: size, weight: .ultraLight))
            .tracking(size * 0.55)
            .foregroundStyle(Palette.text)
            .shadow(color: glow ? Color(hex: 0xBFE0FF).opacity(0.55) : .clear, radius: size * 0.35)
            .accessibilityLabel("AURA")
    }
}

/// Small caps caption with wide tracking ("LESS NOISE. A BRIGHTER YOU.").
struct Caption: View {
    let text: String
    var color: Color = Palette.text3

    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10.5, weight: .medium))
            .tracking(2.6)
            .foregroundStyle(color)
    }
}

struct Screen<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        ZStack {
            LinearGradient(colors: [Palette.bgTop, Palette.bgBottom], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            content
        }
    }
}

/// Frosted slate card with a hairline edge.
struct CardStyle: ViewModifier {
    var padding: CGFloat = 16
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(LinearGradient(colors: [Palette.cardHi.opacity(0.92), Palette.card.opacity(0.92)],
                                         startPoint: .top, endPoint: .bottom))
            )
            .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Palette.stroke, lineWidth: 1))
    }
}

extension View {
    func card(padding: CGFloat = 16) -> some View { modifier(CardStyle(padding: padding)) }
}

extension ListeningMode {
    var symbol: String {
        switch self {
        case .care: "heart.text.square.fill"
        case .focus: "figure.mind.and.body"
        case .commute: "tram.fill"
        case .social: "person.2.fill"
        case .walk: "figure.walk"
        case .custom: "slider.horizontal.3"
        }
    }

    var title: String { ModeProfile.profile(self).title }
    var tagline: String { ModeProfile.profile(self).tagline }

    /// The brand's card titles and handwritten captions.
    var cardTitle: String { self == .focus ? "Café / Study" : title }
    var cardSubtitle: String {
        switch self {
        case .care: "Clearer conversations when the words matter."
        case .focus: "Focus on what fuels you."
        case .commute: "Stay aware. Stay sane."
        case .walk: "More of the world. Less of the noise."
        case .social: "Conversations that count."
        case .custom: "Every sound, your call."
        }
    }
    var script: String {
        switch self {
        case .care: "Every word,\nunderstood."
        case .focus: "Ideas sound\nbetter here."
        case .commute: "Same journey.\nA calmer you."
        case .walk: "Feel more\nout there."
        case .social: "Every word,\ncloser."
        case .custom: "Tuned\nby you."
        }
    }
    var landscape: Landscape.Mood {
        switch self {
        case .care: .golden
        case .focus: .dawn
        case .commute: .dusk
        case .walk: .forest
        case .social: .golden
        case .custom: .mist
        }
    }
}

extension SoundCategory {
    /// Shorter wording for the frosted chips on mode cards.
    func chipLabel(in mode: ListeningMode) -> String {
        switch self {
        case .humanSpeech:
            switch mode {
            case .commute: "Conductor"
            case .social: "Friend's voice"
            case .care: "The person speaking"
            default: "Voices"
            }
        case .approachingVehicle: "Vehicles"
        case .warningSignal: "Warnings"
        case .mechanicalNoise: "Construction"
        default: label(in: mode)
        }
    }
}

extension SoundAction {
    var tint: Color {
        switch self {
        case .enhance: Palette.enhance
        case .preserve: Palette.sky
        case .reduce: Palette.sunrise
        case .suppress: Color(hex: 0x8C96A3)
        }
    }

    var verb: String {
        switch self {
        case .enhance: "Enhanced"
        case .preserve: "Kept"
        case .reduce: "Reduced"
        case .suppress: "Quieted"
        }
    }
}

func formatDB(_ v: Float) -> String {
    let r = (v * 10).rounded() / 10
    if abs(r) < 0.05 { return "0 dB" }
    return String(format: "%+.0f dB", r)
}
