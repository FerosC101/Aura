import AuraCore
import SwiftUI

/// The listening orb: the one element on Home that is alive.
///
/// It answers "is AURA working, and is it hearing me?" without a number, a
/// graph or a dB scale. Three concentric rings breathe at slightly different
/// rates so the motion never looks like a loading spinner, and the whole thing
/// swells with the input level. When Voice Focus holds someone, the palette
/// shifts from lavender to aqua -- the same colour the mixer uses for
/// "brought forward", so the meaning carries across screens.
struct ListeningOrb: View {
    /// Live input level in dBFS, roughly -90 (silence) to 0 (loud).
    var level: Float
    var active: Bool
    var focused: Bool

    @State private var phase: Double = 0

    /// Map dBFS onto 0...1 with a floor, so room tone sits near the bottom of
    /// the range instead of making the orb twitch constantly.
    private var amount: Double {
        guard active else { return 0 }
        let normalised = (Double(level) + 60) / 50
        return min(max(normalised, 0), 1)
    }

    private var tint: Color { focused ? Palette.enhance : Palette.sky }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: !active)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            ZStack {
                ForEach(0..<3, id: \.self) { ring in
                    let offset = Double(ring) * 0.7
                    let breathe = sin(t * (0.5 + offset * 0.18) + offset) * 0.5 + 0.5
                    let scale = 0.72 + breathe * 0.07 + amount * (0.16 - Double(ring) * 0.04)

                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [tint.opacity(0.0), tint.opacity(0.16 - Double(ring) * 0.04)],
                                center: .center, startRadius: 10, endRadius: 150
                            )
                        )
                        .overlay(
                            Circle().stroke(tint.opacity(0.26 - Double(ring) * 0.07), lineWidth: 1)
                        )
                        .scaleEffect(scale)
                }

                // The core: brightest, and the only part that tracks level closely.
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [tint.opacity(0.55), tint.opacity(0.08)],
                            center: .center, startRadius: 2, endRadius: 86
                        )
                    )
                    .scaleEffect(0.42 + amount * 0.13)
                    .blur(radius: 10)

                caption
            }
        }
        // Centred regardless of the stack it lands in: Home lays its content
        // out `.leading`, which otherwise pins the orb to the left margin.
        .frame(maxWidth: .infinity)
        .frame(height: 230)
        .animation(.smooth(duration: 0.5), value: focused)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(active
            ? (focused ? "Voice Focus active" : "AURA is listening")
            : "AURA is paused")
    }

    private var caption: some View {
        VStack(spacing: 6) {
            Text(focused ? "Voice Focus" : active ? "Listening" : "Paused")
                .font(Type_.light(26))
                .foregroundStyle(Palette.text)
            if active {
                Caption(text: focused ? "Holding one voice" : "Selective Awareness", color: tint.opacity(0.85))
            }
        }
    }
}
