import AuraCore
import SwiftUI

struct InsightsView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Caption(text: "Insights")
                    Text("What AURA is doing, right now.").font(Type_.light(24)).foregroundStyle(Palette.text)
                }
                .padding(.top, 14)

                if model.listening == .off {
                    idle
                } else {
                    if model.scene.safetyOverride { SafetyBanner() }
                    nowHearing
                    waveform
                    performance
                }
                sessionCard
                benefits
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .scrollIndicators(.hidden)
    }

    private var idle: some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: "ear.badge.waveform").font(.system(size: 26)).foregroundStyle(Palette.sky)
            Text("AURA isn't listening").font(Type_.medium(16)).foregroundStyle(Palette.text)
            Text("Start listening from Home, or open a demo scene, to see the live acoustic scene and how fast it's processed.")
                .font(Type_.regular(13.5)).foregroundStyle(Palette.text2)
            Button { Task { await model.startLive() } } label: {
                Text("Start listening").font(Type_.semibold(14)).foregroundStyle(Palette.tileSelectedInk)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(Palette.tileSelected, in: Capsule())
            }
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(padding: 20)
    }

    private var nowHearing: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Now hearing").font(Type_.medium(15)).foregroundStyle(Palette.text)
                Spacer()
                if let name = model.lockedName {
                    Label("Person Lock · \(name)", systemImage: "lock.fill")
                        .font(Type_.semibold(11)).foregroundStyle(Palette.enhance)
                }
                if model.scene.nameHeard {
                    Label("Your name", systemImage: "person.wave.2").font(Type_.semibold(11)).foregroundStyle(Palette.sunrise)
                }
            }
            let sounds = model.liveSounds
            if sounds.isEmpty {
                Text("Quiet. Nothing that needs a decision.").font(Type_.regular(13)).foregroundStyle(Palette.text3)
            }
            ForEach(sounds) { s in
                LiveSoundRow(card: s, mode: model.mode, bypassed: !model.enabled || model.comparison != .aura)
            }
        }
        .card()
    }

    private var waveform: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Input vs output").font(Type_.medium(15)).foregroundStyle(Palette.text)
                Spacer()
                Text("in \(Int(model.scene.inputDB)) · out \(Int(model.scene.outputDB)) dB")
                    .font(.system(size: 11, design: .monospaced)).foregroundStyle(Palette.text3)
            }
            WaveformView(input: model.scene.waveIn, output: model.scene.waveOut).frame(height: 76)
            HStack(spacing: 16) {
                legend(Palette.text2, "Microphone")
                legend(Palette.sky, "What you hear")
            }
            bands
        }
        .card()
    }

    private var bands: some View {
        VStack(alignment: .leading, spacing: 6) {
            Caption(text: "Background shaping")
            HStack(alignment: .bottom, spacing: 8) {
                ForEach(Array(BandShaper.bands.enumerated()), id: \.offset) { i, f in
                    let db = i < model.scene.bandDB.count ? model.scene.bandDB[i] : 0
                    VStack(spacing: 4) {
                        ZStack(alignment: .top) {
                            RoundedRectangle(cornerRadius: 3).fill(.white.opacity(0.06)).frame(height: 44)
                            RoundedRectangle(cornerRadius: 3)
                                .fill(db < -1 ? Palette.quiet : Palette.sky.opacity(0.6))
                                .frame(height: max(3, CGFloat(min(abs(db), 24) / 24) * 44))
                        }
                        Text(f >= 1000 ? "\(Int(f / 1000))k" : "\(Int(f))").font(.system(size: 9.5, design: .monospaced)).foregroundStyle(Palette.text3)
                    }
                    .frame(maxWidth: .infinity)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("\(Int(f)) hertz, \(formatDB(db))")
                }
            }
        }
        .padding(.top, 6)
    }

    private func legend(_ c: Color, _ t: String) -> some View {
        HStack(spacing: 6) {
            Capsule().fill(c).frame(width: 10, height: 3)
            Text(t).font(Type_.regular(11)).foregroundStyle(Palette.text3)
        }
    }

    /// End-to-end latency, including the device buffers and the separator's
    /// overlap-add delay -- not just the compute time, which would understate
    /// what the listener experiences by an order of magnitude.
    private var performance: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Performance").font(Type_.medium(15)).foregroundStyle(Palette.text)
            HStack(spacing: 10) {
                metric("Latency", "\(Int(model.totalLatencyMS.rounded()))", "ms", Palette.sky)
                metric("AI load", "\(Int((model.scene.load * 100).rounded()))", "%",
                       model.scene.load > 0.7 ? Palette.safety : Palette.enhance)
                metric("Dropouts", "\(model.scene.dropouts)", "", model.scene.dropouts > 0 ? Palette.sunrise : Palette.text)
            }
            VStack(spacing: 7) {
                stage("Device buffers", AudioController.deviceLatencyMS)
                stage("Frame (\(Int(model.processor.hopMS)) ms)", model.processor.hopMS)
                stage("Separation", model.scene.separationMS)
                stage("Sound recognition", model.scene.classificationMS)
                stage("Policy", model.scene.policyMS)
                stage("Mixing", model.scene.mixingMS)
                stage("Overlap-add delay", model.processor.algorithmicDelayMS)
            }
            HStack {
                Text("\(AudioController.inputName) → \(AudioController.outputName)")
                Spacer()
                Text("\(Int(model.processor.sampleRate / 1000)) kHz")
            }
            .font(Type_.regular(11.5)).foregroundStyle(Palette.text3)
        }
        .card()
    }

    private func metric(_ label: String, _ value: String, _ unit: String, _ tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Caption(text: label)
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.system(size: 22, weight: .semibold, design: .rounded)).foregroundStyle(tint)
                Text(unit).font(Type_.regular(11)).foregroundStyle(Palette.text2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Palette.bgBottom.opacity(0.7), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func stage(_ name: String, _ ms: Float) -> some View {
        let total = max(model.totalLatencyMS, 1)
        return VStack(spacing: 3) {
            HStack {
                Text(name).font(Type_.regular(12)).foregroundStyle(Palette.text2)
                Spacer()
                Text(String(format: ms < 1 ? "%.2f ms" : "%.1f ms", ms)).font(.system(size: 11, design: .monospaced)).foregroundStyle(Palette.text3)
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Capsule().fill(.white.opacity(0.06))
                    Capsule().fill(Palette.skyDeep.opacity(0.8)).frame(width: max(2, g.size.width * CGFloat(ms / total)))
                }
            }
            .frame(height: 3)
        }
    }

    private var sessionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("This session").font(Type_.medium(15)).foregroundStyle(Palette.text)
                Spacer()
                Text("Not saved").font(Type_.regular(11)).foregroundStyle(Palette.text3)
            }
            HStack(spacing: 10) {
                metric("Listening", minutes(model.session.seconds), "min", Palette.text)
                metric("Kept", minutes(model.session.keptSeconds), "min", Palette.sky)
                metric("Quieted", minutes(model.session.quietedSeconds), "min", Palette.quiet)
            }
            if model.session.safetyMoments > 0 {
                Label("\(model.session.safetyMoments) safety override\(model.session.safetyMoments == 1 ? "" : "s") kept a warning audible",
                      systemImage: "shield.fill")
                    .font(Type_.regular(12.5)).foregroundStyle(Palette.safety)
            }
            let top = model.session.quieted.sorted { $0.value > $1.value }.prefix(3)
            if !top.isEmpty {
                Text("Mostly quieted: " + top.map { $0.key.label(in: model.mode) }.joined(separator: ", "))
                    .font(Type_.regular(12.5)).foregroundStyle(Palette.text2)
            }
        }
        .card()
    }

    private func minutes(_ s: Double) -> String { s < 60 ? String(format: "%.1f", s / 60) : "\(Int(s / 60))" }

    private var benefits: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
            benefit("leaf", "Sharper focus", "Work, study, create")
            benefit("face.smiling", "Less stress", "A calmer mind")
            benefit("person.2", "More connection", "Conversations that count")
            benefit("mountain.2", "A more present you", "Everyday, everywhere")
        }
        .padding(.vertical, 10)
    }

    private func benefit(_ symbol: String, _ title: String, _ sub: String) -> some View {
        VStack(spacing: 5) {
            Image(systemName: symbol).font(.system(size: 22, weight: .light)).foregroundStyle(Palette.text2)
            Text(title).font(Type_.medium(13)).foregroundStyle(Palette.text)
            Text(sub).font(Type_.regular(11.5)).foregroundStyle(Palette.text3)
        }
        .frame(maxWidth: .infinity)
    }
}
