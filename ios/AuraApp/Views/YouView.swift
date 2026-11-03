import AuraCore
import SwiftUI

struct YouView: View {
    @Environment(AppModel.self) private var model
    @State private var enrolVoice = false
    @State private var enrolName = false
    @State private var showPrivacy = false

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            Screen {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        VStack(alignment: .leading, spacing: 6) {
                            Caption(text: "You")
                            TextField("Your name", text: $model.profile.userName)
                                .font(Type_.light(26)).foregroundStyle(Palette.text)
                                .textInputAutocapitalization(.words)
                        }
                        .padding(.top, 14)

                        friends
                        nameAwareness
                        safety
                        intensity
                        privacyLink
                        about
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
                }
                .scrollIndicators(.hidden)
            }
            .navigationDestination(isPresented: $showPrivacy) { PrivacyCenterView() }
            .sheet(isPresented: $enrolVoice) { EnrolVoiceView() }
            .sheet(isPresented: $enrolName) { EnrolNameView() }
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    // MARK: Person Lock

    private var friends: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Friend's voice",
                          info: "Person Lock. AURA learns a voice from a few seconds of speech, keeps only a mathematical voiceprint, and lifts that person above everyone else — in Social mode, and whenever you're locked on.")
            if model.speakers.isEmpty {
                Text("Lock onto one person and AURA will bring their voice forward in a crowded room.")
                    .font(Type_.regular(13)).foregroundStyle(Palette.text2)
            }
            ForEach(model.speakers) { p in
                let selected = model.profile.selectedSpeaker == p.id
                HStack(spacing: 12) {
                    ZStack {
                        Circle().fill(selected ? Palette.tileSelected : Palette.bgBottom).frame(width: 36, height: 36)
                        Text(String(p.name.prefix(1))).font(Type_.semibold(15))
                            .foregroundStyle(selected ? Palette.tileSelectedInk : Palette.text)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text(p.name).font(Type_.medium(14.5)).foregroundStyle(Palette.text)
                        Text(model.lockedName == p.name ? "Locked on — speaking now" : selected ? "Listening for their voice" : "Voiceprint saved")
                            .font(Type_.regular(11.5))
                            .foregroundStyle(model.lockedName == p.name ? Palette.enhance : Palette.text3)
                    }
                    Spacer()
                    if !selected {
                        Button("Lock on") { model.selectSpeaker(p.id) }
                            .font(Type_.medium(12.5)).foregroundStyle(Palette.sky)
                    }
                    Menu {
                        Button("Delete voiceprint", role: .destructive) { model.deleteSpeaker(p.id) }
                    } label: {
                        Image(systemName: "ellipsis").foregroundStyle(Palette.text2).frame(width: 30, height: 30)
                    }
                }
            }
            Button { enrolVoice = true } label: {
                Label("Add a voice", systemImage: "plus").font(Type_.medium(14)).foregroundStyle(Palette.skyDeep)
            }
            .padding(.top, 2)
        }
        .card()
    }

    // MARK: Name Awareness

    private var nameAwareness: some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle(isOn: Binding(get: { model.profile.nameAwareness }, set: { model.setNameAwareness($0) })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Name Awareness").font(Type_.medium(15)).foregroundStyle(Palette.text)
                    Text("Lift the voice that says your name. Matched by sound, never transcribed.")
                        .font(Type_.regular(12)).foregroundStyle(Palette.text2)
                }
            }
            .tint(Palette.skyDeep)
            if model.profile.nameAwareness {
                let words = model.profile.nameTemplates.keys.sorted()
                HStack {
                    Text(words.isEmpty ? "Record your name so AURA knows it." : "Listening for “\(words.joined(separator: "”, “"))”")
                        .font(Type_.regular(12.5)).foregroundStyle(Palette.text2)
                    Spacer()
                    Button(words.isEmpty ? "Record" : "Re-record") { enrolName = true }
                        .font(Type_.medium(13)).foregroundStyle(Palette.sky)
                }
                Text("Works best when your name is said on its own — “Vince?” — the way people get your attention.")
                    .font(Type_.regular(11)).foregroundStyle(Palette.text3)
            }
        }
        .card()
    }

    // MARK: Safety

    private var safety: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader(title: "Always audible",
                          info: "These override every mode, slider and learned preference. AURA can make them louder but never takes them below −3 dB, and if it isn't sure what it heard, it keeps the sound.")
            ForEach(SoundCategory.safetyCritical) { c in
                Toggle(isOn: Binding(get: { model.profile.safetySet.contains(c) }, set: { model.setSafety(c, on: $0) })) {
                    HStack(spacing: 12) { Image(systemName: c.info.symbol).frame(width: 24); Text(c.info.label); Image(systemName: "shield.fill").font(.system(size: 9)).foregroundStyle(Palette.safety.opacity(0.85)) }.font(Type_.regular(14)).foregroundStyle(Palette.text)
                }
                .tint(Palette.skyDeep)
            }
            if model.profile.safetySet.count < SoundCategory.safetyCritical.count {
                Text("A sound you turn off here can be quieted like any other. We recommend leaving them all on.")
                    .font(Type_.regular(11.5)).foregroundStyle(Palette.sunrise)
            }
        }
        .card()
    }

    private var intensity: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Suppression intensity").font(Type_.medium(15)).foregroundStyle(Palette.text)
                Spacer()
                Text("\(Int(model.profile.suppression * 100))%").font(.system(size: 13, design: .monospaced)).foregroundStyle(Palette.text2)
            }
            Slider(value: Binding(get: { Double(model.profile.suppression) }, set: { model.setSuppression(Float($0)) }), in: 0.3...1.5)
                .tint(Palette.sky)
            HStack { Text("Gentle"); Spacer(); Text("Deep quiet") }
                .font(Type_.regular(11)).foregroundStyle(Palette.text3)
            let learned = model.profile.learner.records.count
            if learned > 0 {
                HStack {
                    Text("AURA has learned from \(learned) of your adjustments.").font(Type_.regular(12)).foregroundStyle(Palette.text2)
                    Spacer()
                    Button("Reset") { model.forgetLearning() }.font(Type_.medium(12)).foregroundStyle(Palette.sky)
                }
            }
        }
        .card()
    }

    private var privacyLink: some View {
        Button { showPrivacy = true } label: {
            HStack(spacing: 12) {
                Image(systemName: "lock.shield").font(.system(size: 20)).foregroundStyle(Palette.sky)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Privacy Center").font(Type_.medium(15)).foregroundStyle(Palette.text)
                    Text("What AURA hears, keeps, and never keeps.").font(Type_.regular(12)).foregroundStyle(Palette.text2)
                }
                Spacer()
                Image(systemName: "chevron.right").foregroundStyle(Palette.text3)
            }
            .card()
        }
        .buttonStyle(.plain)
    }

    private var about: some View {
        VStack(spacing: 10) {
            Wordmark(size: 18)
            Text("A quieter,\nbrighter tomorrow").font(Type_.script(22)).multilineTextAlignment(.center)
                .foregroundStyle(Palette.text2)
            Caption(text: "AI for a more human tomorrow")
            Text("People  /  Places  /  What matters").font(Type_.regular(10.5)).tracking(2).foregroundStyle(Palette.text3)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
    }
}

// MARK: - Enrolment

/// Records a friend's voice twice, turns it into a voiceprint, and discards
/// the audio. Consent is required and stated plainly.
struct EnrolVoiceView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var consent = false
    @State private var takes: [[Float]] = []
    @State private var recording = false
    @State private var level: Float = 0
    @State private var error: String?
    private let prompts = [
        "“The morning bus was late again, so I walked to the station.”",
        "“Let’s meet by the fountain after class and grab something to eat.”",
    ]

    var body: some View {
        Screen {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Button("Cancel") { dismiss() }.foregroundStyle(Palette.text2)
                    Spacer()
                }
                .padding(.top, 18)
                Text("Add a voice").font(Type_.light(28)).foregroundStyle(Palette.text)
                TextField("Their name", text: $name)
                    .font(Type_.regular(17)).padding(14)
                    .background(Palette.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .foregroundStyle(Palette.text)
                Toggle(isOn: $consent) {
                    Text("\(name.isEmpty ? "This person" : name) knows and agrees to AURA recognising their voice.")
                        .font(Type_.regular(13.5)).foregroundStyle(Palette.text)
                }
                .tint(Palette.skyDeep)

                VStack(alignment: .leading, spacing: 10) {
                    Caption(text: "Take \(min(takes.count + 1, 2)) of 2")
                    Text(prompts[min(takes.count, 1)]).font(Type_.regular(16)).foregroundStyle(Palette.text)
                    Text("Ask them to read this aloud, about an arm's length from your iPhone.")
                        .font(Type_.regular(12.5)).foregroundStyle(Palette.text2)
                    GeometryReader { g in
                        Capsule().fill(.white.opacity(0.08))
                            .overlay(alignment: .leading) {
                                Capsule().fill(Palette.sky).frame(width: g.size.width * CGFloat(min(level * 3, 1)))
                            }
                    }
                    .frame(height: 6)
                }
                .card()

                if let error { Text(error).font(Type_.regular(13)).foregroundStyle(Palette.safety) }

                Spacer()

                Text("Recordings are turned into a voiceprint on this iPhone and discarded immediately. Only the voiceprint is saved, and you can delete it any time.")
                    .font(Type_.regular(11.5)).foregroundStyle(Palette.text3)

                Button { Task { await record() } } label: {
                    Text(recording ? "Listening…" : takes.count == 0 ? "Record" : "Record take 2")
                        .font(Type_.semibold(16)).foregroundStyle(Palette.tileSelectedInk)
                        .frame(maxWidth: .infinity).padding(.vertical, 15)
                        .background(Palette.tileSelected, in: Capsule())
                }
                .disabled(recording || !consent || name.trimmingCharacters(in: .whitespaces).isEmpty)
                .opacity(recording || !consent || name.isEmpty ? 0.5 : 1)
            }
            .padding(.horizontal, 22)
            .padding(.bottom, 24)
        }
    }

    private func record() async {
        error = nil
        recording = true
        let meter = Task { while !Task.isCancelled { level = model.audio.recordingLevel; try? await Task.sleep(for: .milliseconds(60)) } }
        defer { meter.cancel(); recording = false; level = 0 }
        do {
            if AudioController.micPermission == .undetermined { _ = await AudioController.requestMic() }
            takes.append(try await model.audio.record(seconds: 3.5))
            if takes.count == 2 {
                try model.enrolSpeaker(name: name.trimmingCharacters(in: .whitespaces), clips: takes)
                takes = []   // drop the audio: only the voiceprint remains
                model.flash("\(name)'s voiceprint saved")
                dismiss()
            }
        } catch {
            takes = []
            self.error = error.localizedDescription
        }
    }
}

struct EnrolNameView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var takes: [[Float]] = []
    @State private var recording = false
    @State private var error: String?

    private var name: String { model.profile.userName.isEmpty ? "your name" : model.profile.userName }

    var body: some View {
        Screen {
            VStack(alignment: .leading, spacing: 18) {
                HStack { Button("Cancel") { dismiss() }.foregroundStyle(Palette.text2); Spacer() }.padding(.top, 18)
                Text("Teach AURA \(name)").font(Type_.light(28)).foregroundStyle(Palette.text)
                Text("Say “\(name)” on its own, the way a friend calling you would. Twice.")
                    .font(Type_.regular(15)).foregroundStyle(Palette.text2)
                HStack(spacing: 8) {
                    ForEach(0..<2, id: \.self) { i in
                        Capsule().fill(i < takes.count ? Palette.sky : .white.opacity(0.1)).frame(height: 5)
                    }
                }
                if let error { Text(error).font(Type_.regular(13)).foregroundStyle(Palette.safety) }
                Spacer()
                Text("AURA keeps a sound pattern, not a recording, and never transcribes anything.")
                    .font(Type_.regular(11.5)).foregroundStyle(Palette.text3)
                Button { Task { await record() } } label: {
                    Text(recording ? "Listening…" : "Record")
                        .font(Type_.semibold(16)).foregroundStyle(Palette.tileSelectedInk)
                        .frame(maxWidth: .infinity).padding(.vertical, 15)
                        .background(Palette.tileSelected, in: Capsule())
                }
                .disabled(recording)
            }
            .padding(.horizontal, 22).padding(.bottom, 24)
        }
    }

    private func record() async {
        recording = true
        defer { recording = false }
        do {
            takes.append(try await model.audio.record(seconds: 1.4))
            if takes.count == 2 {
                let word = model.profile.userName.isEmpty ? "Name" : model.profile.userName
                if model.enrolName(word, clips: takes) { model.flash("AURA will listen for \(word)"); dismiss() }
                else { error = "Couldn't hear a clear name. Try somewhere quieter." }
                takes = []
            }
        } catch {
            takes = []
            self.error = error.localizedDescription
        }
    }
}
