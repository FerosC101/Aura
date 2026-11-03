import AuraCore
import SwiftUI

/// What AURA processes, what it stores, what it never stores -- and a way to
/// delete all of it.
struct PrivacyCenterView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var confirmDelete = false

    var body: some View {
        Screen {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Button { dismiss() } label: {
                        Label("You", systemImage: "chevron.left").font(Type_.medium(15)).foregroundStyle(Palette.sky)
                    }
                    .padding(.top, 14)
                    VStack(alignment: .leading, spacing: 6) {
                        Caption(text: "Privacy Center")
                        Text("Your world stays on your iPhone.").font(Type_.light(26)).foregroundStyle(Palette.text)
                        Text("AURA processes environmental audio in real time. Raw audio is never stored and never leaves this device.")
                            .font(Type_.regular(14)).foregroundStyle(Palette.text2)
                    }

                    list("What AURA processes", Palette.sky, [
                        "The sound around you, analysed on this iPhone 50 times a second and discarded immediately.",
                        "Which kinds of sound are present, and how confident AURA is — held in memory only.",
                        "How closely speech matches a saved voiceprint, while Person Lock is on.",
                    ])
                    list("What AURA keeps", Palette.sunrise, [
                        "Your listening modes, slider positions and safety choices.",
                        "What it has learned from your adjustments.",
                        "Voiceprints for people who agreed — numbers describing a voice, not recordings.",
                        "The sound pattern of your name, if Name Awareness is on.",
                    ])
                    list("What AURA never keeps", Palette.enhance, [
                        "Recordings of any kind.",
                        "Transcripts of anything said around you.",
                        "A history of where you were or what you heard.",
                    ])

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Microphone").font(Type_.medium(15)).foregroundStyle(Palette.text)
                        Text(micStatus).font(Type_.regular(13)).foregroundStyle(Palette.text2)
                        Button("Open Settings") {
                            if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                        }
                        .font(Type_.medium(13)).foregroundStyle(Palette.sky)
                    }
                    .card()

                    VStack(alignment: .leading, spacing: 10) {
                        Text("Saved voiceprints").font(Type_.medium(15)).foregroundStyle(Palette.text)
                        if model.speakers.isEmpty {
                            Text("None.").font(Type_.regular(13)).foregroundStyle(Palette.text3)
                        }
                        ForEach(model.speakers) { p in
                            HStack {
                                Text(p.name).font(Type_.regular(14)).foregroundStyle(Palette.text)
                                Spacer()
                                Text("\(p.centroid.count) numbers · \(p.created.formatted(date: .abbreviated, time: .omitted))")
                                    .font(Type_.regular(11.5)).foregroundStyle(Palette.text3)
                                Button { model.deleteSpeaker(p.id) } label: {
                                    Image(systemName: "trash").foregroundStyle(Palette.safety)
                                }
                                .accessibilityLabel("Delete \(p.name)'s voiceprint")
                            }
                        }
                    }
                    .card()

                    Toggle(isOn: Binding(get: { model.profile.sessionInsights }, set: { model.profile.sessionInsights = $0 })) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Session insights").font(Type_.medium(15)).foregroundStyle(Palette.text)
                            Text("Count how long sounds were kept or quieted. In memory only; gone when you close AURA.")
                                .font(Type_.regular(12)).foregroundStyle(Palette.text2)
                        }
                    }
                    .tint(Palette.skyDeep)
                    .card()

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Cloud processing").font(Type_.medium(15)).foregroundStyle(Palette.text)
                        Text("Off, and not built. AURA has no code that uploads audio. If it ever offers cloud features, they will be separate and opt-in.")
                            .font(Type_.regular(12.5)).foregroundStyle(Palette.text2)
                    }
                    .card()

                    Button(role: .destructive) { confirmDelete = true } label: {
                        Text("Delete everything AURA has stored")
                            .font(Type_.semibold(15)).foregroundStyle(Palette.safety)
                            .frame(maxWidth: .infinity).padding(.vertical, 15)
                            .background(Palette.safety.opacity(0.12), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                    .confirmationDialog("Delete everything?", isPresented: $confirmDelete, titleVisibility: .visible) {
                        Button("Delete voiceprints, preferences and learning", role: .destructive) { model.deleteEverything() }
                    } message: {
                        Text("This can't be undone.")
                    }
                }
                .padding(.horizontal, 20).padding(.bottom, 20)
            }
            .scrollIndicators(.hidden)
        }
        .toolbar(.hidden, for: .navigationBar)
    }

    private var micStatus: String {
        switch AudioController.micPermission {
        case .granted: "Allowed. AURA only uses it while listening or recording a voice."
        case .denied: "Not allowed. AURA can still play demo scenes."
        default: "Not asked yet."
        }
    }

    private func list(_ title: String, _ tint: Color, _ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(Type_.medium(15)).foregroundStyle(tint)
            ForEach(items, id: \.self) { i in
                HStack(alignment: .top, spacing: 10) {
                    Circle().fill(tint.opacity(0.7)).frame(width: 5, height: 5).padding(.top, 7)
                    Text(i).font(Type_.regular(13)).foregroundStyle(Palette.text2)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}
