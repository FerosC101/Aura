import AuraCore
import SwiftUI

/// Voice Focus: pick the person you are talking to, and AURA holds them above
/// everything else.
///
/// The engine feature underneath is Person Lock, which has existed since the
/// first build. What was missing was a way to *use* it in the moment: it lived
/// in a settings screen, next to enrolment. Someone sitting across from a
/// pharmacist is not going to open settings. So this is a sheet reachable in
/// one tap from Home, and it asks exactly one question.
struct VoiceFocusView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    /// A short listening beat before the list appears. The engine is already
    /// running and needs no warm-up -- this exists because choosing a voice
    /// out of silence is confusing, and a moment of "listening" makes it clear
    /// that the names below are people AURA can actually hear.
    @State private var listening = true

    private var speakers: [SpeakerProfile] { model.profile.speakers }

    var body: some View {
        NavigationStack {
            Screen {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        intro
                        if listening && !speakers.isEmpty {
                            listeningState
                        } else if speakers.isEmpty {
                            emptyState
                        } else {
                            speakerList
                            intensity
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 28)
                }
                .scrollIndicators(.hidden)
            }
            .navigationTitle("Voice Focus")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }.foregroundStyle(Palette.sky)
                }
            }
        }
        .task {
            try? await Task.sleep(for: .milliseconds(1400))
            withAnimation(.smooth(duration: 0.4)) { listening = false }
        }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Who do you want to hear?")
                .font(Type_.regular(24))
                .foregroundStyle(Palette.text)
            Text("AURA will hold that voice above the room and step everything else back.")
                .font(Type_.regular(14))
                .foregroundStyle(Palette.text2)
        }
        .padding(.top, 8)
    }

    private var listeningState: some View {
        VStack(spacing: 14) {
            ListeningOrb(level: model.scene.inputDB, active: true, focused: false)
            Text("Listening…")
                .font(Type_.regular(15))
                .foregroundStyle(Palette.text2)
        }
        .frame(maxWidth: .infinity)
        .transition(.opacity)
    }

    private var speakerList: some View {
        VStack(spacing: 2) {
            ForEach(speakers) { speaker in
                let selected = model.profile.selectedSpeaker == speaker.id
                Button {
                    model.selectSpeaker(selected ? nil : speaker.id)
                } label: {
                    HStack(spacing: 14) {
                        ZStack {
                            Circle()
                                .fill(selected ? Palette.enhance.opacity(0.18) : Palette.cardHi)
                                .frame(width: 44, height: 44)
                            Image(systemName: selected ? "waveform" : "person.wave.2")
                                .font(.system(size: 17))
                                .foregroundStyle(selected ? Palette.enhance : Palette.text2)
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            Text(speaker.name)
                                .font(Type_.medium(16))
                                .foregroundStyle(Palette.text)
                            Text(selected ? "Held above the room" : "Tap to focus")
                                .font(Type_.regular(13))
                                .foregroundStyle(selected ? Palette.enhance : Palette.text3)
                        }
                        Spacer()
                        if selected {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 20))
                                .foregroundStyle(Palette.enhance)
                        }
                    }
                    .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
                if speaker.id != speakers.last?.id { Divider().overlay(Palette.stroke) }
            }
        }
        .card()
    }

    /// One control, in words rather than decibels. The spec is explicit that
    /// nobody should be asked to manage frequency bands to be understood.
    private var intensity: some View {
        @Bindable var model = model
        return VStack(alignment: .leading, spacing: 10) {
            SectionHeader(title: "How much?",
                          info: "Natural keeps more of the room around you. Clearer pushes it further back so the voice carries. Safety sounds stay audible at either end.")
            Slider(value: $model.profile.suppression, in: 0.4...1.6)
                .tint(Palette.skyDeep)
            HStack {
                Text("Natural").font(Type_.regular(13)).foregroundStyle(Palette.text3)
                Spacer()
                Text("Clearer").font(Type_.regular(13)).foregroundStyle(Palette.text3)
            }
        }
        .card()
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("No voices saved yet")
                .font(Type_.medium(16))
                .foregroundStyle(Palette.text)
            Text("Add the people you talk with often — a doctor, a pharmacist, "
                 + "someone at home — and AURA can hold their voice above a noisy room. "
                 + "AURA stores a voice signature, never a recording.")
                .font(Type_.regular(14))
                .foregroundStyle(Palette.text2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }
}
