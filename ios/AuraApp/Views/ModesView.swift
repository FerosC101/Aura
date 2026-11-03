import AuraCore
import SwiftUI

struct ModesView: View {
    @Environment(AppModel.self) private var model
    @State private var showDemo = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Caption(text: "Listening modes")
                    Text("Designed for a more present tomorrow.")
                        .font(Type_.light(24)).foregroundStyle(Palette.text)
                }
                .padding(.top, 14)

                ForEach([ListeningMode.care, .focus, .commute, .walk, .social, .custom]) { m in
                    ModeCard(mode: m, selected: model.mode == m) { model.setMode(m) }
                }

                Button { showDemo = true } label: {
                    HStack(spacing: 14) {
                        Image(systemName: "waveform.badge.magnifyingglass").font(.system(size: 22)).foregroundStyle(Palette.sky)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Hear the difference").font(Type_.medium(15)).foregroundStyle(Palette.text)
                            Text("Five scripted scenes. Raw, noise cancelling, or AURA.")
                                .font(Type_.regular(12.5)).foregroundStyle(Palette.text2)
                        }
                        Spacer()
                        Image(systemName: "chevron.right").foregroundStyle(Palette.text3)
                    }
                    .card()
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .scrollIndicators(.hidden)
        .sheet(isPresented: $showDemo) { DemoView() }
    }
}

struct ModeCard: View {
    @Environment(AppModel.self) private var model
    let mode: ListeningMode
    let selected: Bool
    let action: () -> Void

    /// The chips mirror what the mode actually does, straight from the policy.
    private var chips: [(SoundCategory, Bool)] {
        model.controlCategories(mode).prefix(4).map { c in
            (c, model.decisionFor(c, mode: mode).gainDB >= -2)
        }
    }

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topLeading) {
                Landscape(mood: mode.landscape, showLake: mode != .walk)
                    .overlay(LinearGradient(colors: [.black.opacity(0.45), .clear, .black.opacity(0.4)],
                                            startPoint: .top, endPoint: .bottom))
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(mode.cardTitle).font(Type_.semibold(19))
                        Spacer()
                        if selected {
                            Text("Active").font(Type_.semibold(11)).foregroundStyle(Palette.tileSelectedInk)
                                .padding(.horizontal, 10).padding(.vertical, 4)
                                .background(Palette.tileSelected, in: Capsule())
                        }
                    }
                    Text(mode.cardSubtitle).font(Type_.regular(13.5)).opacity(0.9)
                }
                .foregroundStyle(.white)
                .padding(16)

                HStack(alignment: .bottom) {
                    Text(mode.script)
                        .font(Type_.script(21))
                        .fixedSize()
                        .foregroundStyle(.white.opacity(0.92))
                        .rotationEffect(.degrees(-6))
                        .shadow(color: .black.opacity(0.35), radius: 6)
                    Spacer()
                    VStack(spacing: 6) {
                        ForEach(chips, id: \.0) { c, plus in
                            SoundChip(symbol: c.info.symbol, label: c.chipLabel(in: mode), plus: plus)
                        }
                    }
                }
                .padding(14)
                .frame(maxHeight: .infinity, alignment: .bottom)
            }
            .frame(height: 250)
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(selected ? Palette.sky.opacity(0.7) : Palette.stroke, lineWidth: selected ? 1.5 : 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(mode.title) mode. \(mode.cardSubtitle)")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

extension AppModel {
    func decisionFor(_ c: SoundCategory, mode m: ListeningMode) -> GainDecision {
        let engine = PolicyEngine()
        engine.safety.enabled = profile.safetySet
        return engine.explain(c, policyContext(for: m))
    }
}
