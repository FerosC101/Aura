import AuraCore
import SwiftUI

/// PSC Demo Mode. The same scene through three chains -- raw, conventional
/// noise cancelling, and AURA -- so the difference can be heard and seen on
/// identical material.
struct DemoView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var selected: DemoScene? = DemoCatalog.scenes.first

    private var playingKey: String? {
        if case .demo(let k) = model.listening { k } else { nil }
    }

    var body: some View {
        Screen {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        Wordmark(size: 16)
                        Spacer()
                        Button("Done") { dismiss() }.font(Type_.medium(15)).foregroundStyle(Palette.sky)
                    }
                    .padding(.top, 18)

                    VStack(alignment: .leading, spacing: 6) {
                        Caption(text: "PSC Demo Mode")
                        Text("Hear the difference.").font(Type_.light(28)).foregroundStyle(Palette.text)
                        Text("Noise cancellation blocks the world. AURA learns what parts of it matter.")
                            .font(Type_.regular(14)).foregroundStyle(Palette.text2)
                    }

                    ScrollView(.horizontal) {
                        HStack(spacing: 10) {
                            ForEach(DemoCatalog.scenes) { s in
                                Button { select(s) } label: {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(s.title).font(Type_.medium(13.5))
                                        Text(s.subtitle).font(Type_.regular(11)).opacity(0.7)
                                    }
                                    .foregroundStyle(selected == s ? Palette.tileSelectedInk : Palette.text)
                                    .padding(.horizontal, 14).padding(.vertical, 10)
                                    .background(selected == s ? AnyShapeStyle(Palette.tileSelected) : AnyShapeStyle(Palette.card),
                                                in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .scrollIndicators(.hidden)

                    comparisonPicker

                    if let s = selected { sceneCard(s) }

                    liveCard
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 30)
            }
            .scrollIndicators(.hidden)
        }
    }

    private func select(_ s: DemoScene) {
        selected = s
        if playingKey != nil { model.startDemo(s) }
    }

    private var comparisonPicker: some View {
        HStack(spacing: 4) {
            ForEach(Comparison.allCases) { c in
                Button { model.setComparison(c) } label: {
                    Text(c.title)
                        .font(Type_.medium(13))
                        .foregroundStyle(model.comparison == c ? Palette.text : Palette.text2)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(model.comparison == c ? Palette.cardHi : .clear, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(Palette.bgBottom, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 13, style: .continuous).stroke(Palette.stroke))
        .overlay(alignment: .bottom) {
            Text(model.comparison.explainer(for: selected?.listeningMode ?? .commute))
                .font(Type_.regular(12)).foregroundStyle(Palette.text3)
                .multilineTextAlignment(.center).offset(y: 34)
        }
        .padding(.bottom, 26)
    }

    private func sceneCard(_ s: DemoScene) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(s.title).font(Type_.semibold(16)).foregroundStyle(Palette.text)
                    Text("\(s.listeningMode.title) mode · \(Int(s.duration)) s loop").font(Type_.regular(12)).foregroundStyle(Palette.text2)
                }
                Spacer()
                Button {
                    playingKey == s.key ? model.stop() : model.startDemo(s)
                } label: {
                    Image(systemName: playingKey == s.key ? "stop.fill" : "play.fill")
                        .font(.system(size: 16)).foregroundStyle(Palette.tileSelectedInk)
                        .frame(width: 46, height: 46).background(Palette.tileSelected, in: Circle())
                }
                .accessibilityLabel(playingKey == s.key ? "Stop" : "Play \(s.title)")
            }
            Text(s.description).font(Type_.regular(13)).foregroundStyle(Palette.text2)
            Divider().overlay(Palette.stroke)
            ForEach(Array(s.layers.enumerated()), id: \.offset) { _, l in
                let cat = SoundCategory.from(key: l.category)
                let want = s.expect[l.category].flatMap(SoundAction.init(rawValue:))
                HStack(spacing: 10) {
                    Image(systemName: cat?.symbol(in: s.listeningMode) ?? "waveform").frame(width: 20).foregroundStyle(Palette.text2)
                    Text(cat?.label(in: s.listeningMode) ?? l.category).font(Type_.regular(13.5)).foregroundStyle(Palette.text)
                    Spacer()
                    Text(String(format: "%.0f–%.0f s", l.start, l.end)).font(.system(size: 11, design: .monospaced)).foregroundStyle(Palette.text3)
                    if let want {
                        Text(want.verb).font(Type_.semibold(11)).foregroundStyle(want.tint).frame(width: 66, alignment: .trailing)
                    }
                }
            }
        }
        .card()
    }

    private var liveCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("What AURA hears").font(Type_.medium(15)).foregroundStyle(Palette.text)
                Spacer()
                if model.scene.safetyOverride {
                    Label("Safety override", systemImage: "shield.fill").font(Type_.semibold(11)).foregroundStyle(Palette.safety)
                }
            }
            if playingKey == nil {
                Text("Press play. The scene loops; switch between Raw, Noise cancelling and AURA while it runs.")
                    .font(Type_.regular(13)).foregroundStyle(Palette.text3)
            } else {
                WaveformView(input: model.scene.waveIn, output: model.scene.waveOut).frame(height: 64)
                ForEach(model.liveSounds) { s in
                    LiveSoundRow(card: s, mode: model.mode, bypassed: model.comparison != .aura)
                }
            }
        }
        .card()
    }
}

extension Comparison {
    var title: String {
        switch self {
        case .raw: "Raw"
        case .anc: "Noise cancelling"
        case .aura: "AURA"
        }
    }
    /// Worded for the scene on screen. The AURA line is the one that has to
    /// change: "the horn and the siren stay" means nothing in a consultation
    /// room, and a judge reading it there learns that the copy is generic.
    func explainer(for mode: ListeningMode) -> String {
        switch self {
        case .raw: return "Everything, exactly as the microphone hears it."
        case .anc: return mode == .care
            ? "Quieter by frequency — and the voice goes down with the room."
            : "Quieter by frequency — and the siren goes with the engine."
        case .aura: return mode == .care
            ? "The person speaking comes forward. The room steps back. The alarm is untouched."
            : "The speaker, the horn and the siren stay. The rest steps back."
        }
    }
}

/// A detected sound with what AURA is doing about it.
struct LiveSoundRow: View {
    let card: SoundCard
    let mode: ListeningMode
    var bypassed = false

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: card.category.symbol(in: mode))
                .font(.system(size: 14)).foregroundStyle(tint)
                .frame(width: 32, height: 32)
                .background(Palette.bgBottom, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 4) {
                Text(card.category.label(in: mode)).font(Type_.regular(14)).foregroundStyle(Palette.text)
                GeometryReader { g in
                    ZStack(alignment: .leading) {
                        Capsule().fill(.white.opacity(0.07))
                        Capsule().fill(tint).frame(width: g.size.width * CGFloat(card.confidence))
                    }
                }
                .frame(height: 3)
            }
            VStack(alignment: .trailing, spacing: 2) {
                Text(bypassed ? "Passed" : card.action.verb).font(Type_.semibold(11)).tracking(0.6).foregroundStyle(tint)
                Text(bypassed ? "0 dB" : formatDB(card.gainDB)).font(.system(size: 11.5, design: .monospaced)).foregroundStyle(Palette.text2)
                if card.safetyOverride && !bypassed {
                    Text("SAFETY").font(.system(size: 8.5, weight: .bold)).tracking(1).foregroundStyle(Palette.safety)
                }
            }
            .frame(width: 74, alignment: .trailing)
        }
        .frame(height: 40)
        .animation(.smooth(duration: 0.25), value: card.confidence)
    }

    private var tint: Color { bypassed ? Palette.text2 : card.action.tint }
}
