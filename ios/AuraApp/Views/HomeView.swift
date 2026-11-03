import AuraCore
import SwiftUI

struct HomeView: View {
    @Environment(AppModel.self) private var model
    var openTab: (RootView.Tab) -> Void
    @State private var showDemo = false
    @State private var showVoiceFocus = LaunchOptions.sheet == "voicefocus"

    private var greeting: String {
        let h = Calendar.current.component(.hour, from: Date())
        let part = h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening"
        return model.profile.userName.isEmpty ? "\(part)," : "\(part), \(model.profile.userName)"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                VStack(alignment: .leading, spacing: 4) {
                    Text(greeting).font(Type_.regular(14)).foregroundStyle(Palette.text2)
                    Text("Understand the voice that matters.")
                        .font(Type_.regular(24))
                        .foregroundStyle(Palette.text)
                        .contentTransition(.opacity)
                }
                .padding(.top, 4)

                status

                ListeningOrb(level: model.scene.inputDB,
                             active: model.listening != .off,
                             focused: model.scene.lock.active)

                voiceFocusButton

                HStack(spacing: 10) {
                    ForEach([ListeningMode.care, .commute, .focus, .social]) { m in
                        ModeTile(mode: m, selected: model.mode == m) { model.setMode(m) }
                    }
                }

                HeroCard(showDemo: $showDemo)

                if model.scene.safetyOverride { SafetyBanner().transition(.opacity) }

                controlCard

                Caption(text: "ANC controls noise. AURA controls relevance.")
                    .frame(maxWidth: .infinity)
                    .padding(.top, 6)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .scrollIndicators(.hidden)
        .animation(.smooth(duration: 0.3), value: model.scene.safetyOverride)
        .sheet(isPresented: $showDemo) { DemoView() }
        .sheet(isPresented: $showVoiceFocus) { VoiceFocusView() }
    }

    /// What AURA is doing right now, in two lines and no numbers.
    private var status: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(model.listening != .off ? Palette.enhance : Palette.text3)
                .frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 1) {
                Text("Selective Awareness")
                    .font(Type_.medium(15))
                    .foregroundStyle(Palette.text)
                Text(model.listening != .off
                     ? "Helping you focus on the sounds that matter."
                     : "Paused. Tap below to start listening.")
                    .font(Type_.regular(13))
                    .foregroundStyle(Palette.text2)
            }
            Spacer()
            Text(model.listening != .off ? "Active" : "Off")
                .font(Type_.medium(13))
                .foregroundStyle(model.listening != .off ? Palette.enhance : Palette.text3)
        }
        .card(padding: 14)
    }

    /// The primary action. Voice Focus is the flagship feature, so it gets the
    /// only filled button on the screen.
    private var voiceFocusButton: some View {
        Button { showVoiceFocus = true } label: {
            HStack(spacing: 10) {
                Image(systemName: "person.wave.2.fill").font(.system(size: 16))
                Text(model.scene.lock.active
                     ? "Voice Focus - \(model.scene.lock.name ?? "active")"
                     : "Focus on a Voice")
                    .font(Type_.medium(17))
            }
            .foregroundStyle(Palette.tileSelectedInk)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(RoundedRectangle(cornerRadius: 18, style: .continuous).fill(Palette.tileSelected))
        }
        .buttonStyle(.plain)
    }

    private var header: some View {
        ZStack {
            Wordmark(size: 21)
            HStack {
                Spacer()
                Button { openTab(.you) } label: {
                    Image(systemName: "gearshape").font(.system(size: 18)).foregroundStyle(Palette.text2)
                }
                .accessibilityLabel("Settings")
            }
        }
        .padding(.top, 8)
    }

    private var controlCard: some View {
        VStack(spacing: 2) {
            SectionHeader(title: "Selective Sound Control",
                          info: "Plus lets a sound through, minus turns it down. Drag a bar for finer control. Sounds with a shield always stay audible, whatever you set.")
                .padding(.bottom, 6)
            ForEach(model.controlCategories(model.mode)) { c in
                SoundControlRow(category: c)
                if c != model.controlCategories(model.mode).last {
                    Divider().overlay(Palette.stroke)
                }
            }
        }
        .card()
    }
}

/// The landscape card from the brand sheet, doing real work: it is the master
/// control and the live status line.
struct HeroCard: View {
    @Environment(AppModel.self) private var model
    @Binding var showDemo: Bool

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            Landscape(mood: model.mode.landscape)
                .overlay(LinearGradient(colors: [.clear, .black.opacity(0.35)], startPoint: .center, endPoint: .bottom))

            VStack(spacing: 4) {
                Text("Quieter surroundings.")
                Text("A brighter you.")
            }
            .font(Type_.regular(16))
            .foregroundStyle(Color(hex: 0x2B4764))
            .frame(maxWidth: .infinity)
            .padding(.top, 26)
            .frame(maxHeight: .infinity, alignment: .top)

            VStack(alignment: .leading, spacing: 10) {
                statusLine
                controls
            }
            .padding(14)
        }
        .frame(height: 214)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Palette.stroke))
    }

    @ViewBuilder private var statusLine: some View {
        switch model.listening {
        case .off:
            EmptyView()
        case .live, .demo:
            HStack(spacing: 8) {
                Circle().fill(model.enabled ? Palette.enhance : Palette.sunrise).frame(width: 7, height: 7)
                    .shadow(color: Palette.enhance, radius: model.enabled ? 4 : 0)
                Text(model.enabled ? "Selective Awareness Active" : "AURA paused — hearing everything")
                    .font(Type_.medium(12.5))
                Spacer()
                Text("\(Int(model.totalLatencyMS.rounded())) ms").font(.system(size: 11.5, weight: .medium, design: .monospaced))
                    .opacity(0.8)
            }
            .foregroundStyle(.white)
        }
    }

    private var controls: some View {
        HStack(spacing: 10) {
            switch model.listening {
            case .off:
                Button { Task { await model.startLive() } } label: {
                    Label("Start listening", systemImage: "ear")
                        .font(Type_.semibold(14))
                        .foregroundStyle(Palette.tileSelectedInk)
                        .padding(.horizontal, 16).padding(.vertical, 11)
                        .background(Palette.tileSelected, in: Capsule())
                }
                Button { showDemo = true } label: {
                    Text("Hear the difference")
                        .font(Type_.medium(13.5))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14).padding(.vertical, 11)
                        .background(.ultraThinMaterial, in: Capsule())
                }
            case .live, .demo:
                @Bindable var model = model
                Toggle(isOn: $model.enabled) {
                    Text("AURA").font(.system(size: 13, weight: .semibold)).tracking(2.5).foregroundStyle(.white)
                }
                .toggleStyle(.switch)
                .tint(Palette.skyDeep)
                .fixedSize()
                .padding(.leading, 14).padding(.trailing, 8).padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                Spacer()
                Button { model.stop() } label: {
                    Image(systemName: "stop.fill").font(.system(size: 13)).foregroundStyle(.white)
                        .frame(width: 38, height: 38).background(.ultraThinMaterial, in: Circle())
                }
                .accessibilityLabel("Stop listening")
            }
        }
    }
}
