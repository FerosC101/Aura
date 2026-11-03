import AuraCore
import SwiftUI

/// The six onboarding screens from the brief.
struct OnboardingView: View {
    @Environment(AppModel.self) private var model
    @State private var step = LaunchOptions.onboardingStep ?? 0
    @State private var uses: Set<String> = []
    @State private var safety: Set<SoundCategory> = Set(SoundCategory.safetyCritical)
    @State private var name = ""
    @State private var enrolVoice = false

    private let useOptions: [(String, String, ListeningMode)] = [
        ("Studying", "book", .focus), ("Commuting", "tram", .commute), ("Walking", "figure.walk", .walk),
        ("Conversations", "person.2", .social), ("Work", "laptopcomputer", .focus),
    ]

    var body: some View {
        ZStack {
            // Sky above, mountains in the lower half: on a full-height canvas
            // the ridges tower behind the copy.
            ZStack(alignment: .bottom) {
                LinearGradient(colors: [Palette.bgTop, Color(hex: 0x3B4A61)], startPoint: .top, endPoint: .bottom)
                Landscape(mood: step < 2 ? .dawn : .mist)
                    .frame(height: 470)
                    // Feather the top edge into the sky gradient: no seam.
                    .mask(LinearGradient(stops: [.init(color: .clear, location: 0), .init(color: .black, location: 0.3)],
                                         startPoint: .top, endPoint: .bottom))
            }
                .ignoresSafeArea()
                .overlay(
                    LinearGradient(stops: step < 2
                        ? [.init(color: Palette.bgBottom.opacity(0.35), location: 0),
                           .init(color: .clear, location: 0.35),
                           .init(color: .clear, location: 0.6),
                           .init(color: Palette.bgBottom.opacity(0.92), location: 0.86)]
                        : [.init(color: Palette.bgBottom.opacity(0.6), location: 0),
                           .init(color: Palette.bgBottom.opacity(0.96), location: 0.5)],
                        startPoint: .top, endPoint: .bottom)
                    .ignoresSafeArea()
                )
            VStack(spacing: 0) {
                Wordmark(size: step == 0 ? 44 : 18, glow: step == 0)
                    .padding(.top, step == 0 ? 90 : 20)
                    .animation(.smooth(duration: 0.5), value: step)
                Group {
                    switch step {
                    case 0: welcome
                    case 1: selective
                    case 2: primaryUses
                    case 3: safetySounds
                    case 4: personLock
                    default: microphone
                    }
                }
                .transition(.asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity), removal: .opacity))
                .frame(maxHeight: .infinity, alignment: .top)
                .padding(.top, 28)

                footer
            }
            .padding(.horizontal, 26)
            .padding(.bottom, 20)
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $enrolVoice) { EnrolVoiceView() }
    }

    private var footer: some View {
        VStack(spacing: 18) {
            HStack(spacing: 6) {
                ForEach(0..<6, id: \.self) { i in
                    Capsule().fill(i == step ? Palette.sky : .white.opacity(0.2)).frame(width: i == step ? 18 : 6, height: 6)
                }
            }
            Button { advance() } label: {
                Text(step == 5 ? "Allow microphone & start" : step == 0 ? "Get started" : "Continue")
                    .font(Type_.semibold(16)).foregroundStyle(Palette.tileSelectedInk)
                    .frame(maxWidth: .infinity).padding(.vertical, 16)
                    .background(Palette.tileSelected, in: Capsule())
            }
            if step > 0 {
                Button("Back") { withAnimation(.smooth) { step -= 1 } }
                    .font(Type_.medium(14)).foregroundStyle(Palette.text2)
            }
        }
        .animation(.smooth, value: step)
    }

    private func advance() {
        if step < 5 { withAnimation(.smooth(duration: 0.4)) { step += 1 }; return }
        Task {
            _ = await AudioController.requestMic()
            model.profile.userName = name.trimmingCharacters(in: .whitespaces)
            model.profile.primaryUses = Array(uses)
            model.profile.safetyEnabled = safety.map(\.info.key)
            if let first = useOptions.first(where: { uses.contains($0.0) }) { model.setMode(first.2) }
            for c in SoundCategory.safetyCritical { model.setSafety(c, on: safety.contains(c)) }
            model.profile.onboarded = true
        }
    }

    // MARK: Screens

    private var welcome: some View {
        VStack(spacing: 22) {
            VStack(spacing: 6) {
                Text("Hear what matters.")
                Text("Silence what doesn’t.")
            }
            .font(Type_.light(24)).foregroundStyle(Palette.text).tracking(0.5)
            .shadow(color: .black.opacity(0.25), radius: 8)
            Text("Noise cancellation blocks the world.\nAURA learns what parts of it matter.")
                .font(Type_.regular(15)).foregroundStyle(Palette.text2).multilineTextAlignment(.center)
            Text("A quieter, brighter tomorrow").font(Type_.script(24)).foregroundStyle(Palette.text.opacity(0.85))
                .rotationEffect(.degrees(-5)).padding(.top, 10)
        }
        .frame(maxWidth: .infinity)
    }

    private var selective: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("Selective Awareness", "AURA hears everything around you at once, works out what each sound is, and builds a different mix for you — in real time.")
            row(.enhance, "Enhanced", "Anything that warns you: horns, sirens, bells")
            row(.preserve, "Kept", "The voice in front of you, announcements, your name")
            row(.reduce, "Reduced", "Traffic, crowds, repetitive background")
            row(.suppress, "Quieted", "Engine rumble, machinery, fans")
        }
    }

    private func row(_ a: SoundAction, _ t: String, _ s: String) -> some View {
        HStack(spacing: 14) {
            Capsule().fill(a.tint).frame(width: 3, height: 36)
            VStack(alignment: .leading, spacing: 2) {
                Text(t).font(Type_.semibold(15)).foregroundStyle(a.tint)
                Text(s).font(Type_.regular(13)).foregroundStyle(Palette.text2)
            }
        }
    }

    private var primaryUses: some View {
        VStack(alignment: .leading, spacing: 16) {
            title("What will you use AURA for?", "Pick any. This sets your starting mode.")
            ForEach(useOptions, id: \.0) { label, symbol, _ in
                let on = uses.contains(label)
                Button { if on { uses.remove(label) } else { uses.insert(label) } } label: {
                    HStack(spacing: 14) {
                        Image(systemName: symbol).frame(width: 24)
                        Text(label).font(Type_.medium(15))
                        Spacer()
                        Image(systemName: on ? "checkmark.circle.fill" : "circle").foregroundStyle(on ? Palette.sky : Palette.text3)
                    }
                    .foregroundStyle(Palette.text)
                    .padding(14)
                    .background(on ? Palette.cardHi : Palette.card.opacity(0.8), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(on ? Palette.sky.opacity(0.5) : Palette.stroke))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var safetySounds: some View {
        VStack(alignment: .leading, spacing: 14) {
            title("Sounds that always get through", "These override every other setting. AURA will never silence them.")
            ForEach(SoundCategory.safetyCritical) { c in
                Toggle(isOn: Binding(get: { safety.contains(c) }, set: { if $0 { safety.insert(c) } else { safety.remove(c) } })) {
                    HStack(spacing: 12) { Image(systemName: c.info.symbol).frame(width: 24); Text(c.info.label); Image(systemName: "shield.fill").font(.system(size: 9)).foregroundStyle(Palette.safety.opacity(0.85)) }.font(Type_.regular(15)).foregroundStyle(Palette.text)
                }
                .tint(Palette.skyDeep)
            }
            if safety.count < SoundCategory.safetyCritical.count {
                Text("A sound you turn off may be quieted. We recommend leaving them all on.")
                    .font(Type_.regular(12)).foregroundStyle(Palette.sunrise)
            }
        }
    }

    private var personLock: some View {
        VStack(alignment: .leading, spacing: 16) {
            title("Friend's voice", "Lock onto one person and AURA lifts them above the room. A short sample becomes a voiceprint; the recording itself is discarded straight away.")
            Button { enrolVoice = true } label: {
                Label(model.speakers.isEmpty ? "Add a voice now" : "Add another voice", systemImage: "person.wave.2")
                    .font(Type_.medium(15)).foregroundStyle(Palette.text)
                    .frame(maxWidth: .infinity).padding(.vertical, 14)
                    .background(Palette.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            ForEach(model.speakers) { p in
                Label(p.name, systemImage: "checkmark.circle.fill").foregroundStyle(Palette.enhance).font(Type_.regular(14))
            }
            Text("Optional — you can do this later from You.").font(Type_.regular(12.5)).foregroundStyle(Palette.text3)
        }
    }

    private var microphone: some View {
        VStack(alignment: .leading, spacing: 18) {
            title("One last thing", "AURA needs your microphone to hear the world around you.")
            VStack(alignment: .leading, spacing: 10) {
                Label("AURA processes environmental audio in real time. Raw audio is not stored by default.", systemImage: "lock.shield")
                    .font(Type_.medium(14)).foregroundStyle(Palette.text)
                Text("Everything runs on this iPhone. Nothing is uploaded. Only your settings are saved.")
                    .font(Type_.regular(13)).foregroundStyle(Palette.text2)
            }
            .card()
            VStack(alignment: .leading, spacing: 6) {
                Caption(text: "Your name (for Name Awareness)")
                TextField("Vince", text: $name)
                    .font(Type_.regular(17)).padding(14)
                    .background(Palette.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
        }
    }

    private func title(_ t: String, _ sub: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(t).font(Type_.light(28)).foregroundStyle(Palette.text)
            Text(sub).font(Type_.regular(14.5)).foregroundStyle(Palette.text2)
        }
    }
}
