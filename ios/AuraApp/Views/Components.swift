import AuraCore
import SwiftUI

/// A mode tile from the Home screen: light when selected, slate otherwise.
struct ModeTile: View {
    let mode: ListeningMode
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: mode.symbol)
                    .font(.system(size: 19, weight: .regular))
                    .frame(height: 22)
                Text(mode.title)
                    .font(Type_.medium(12))
            }
            .foregroundStyle(selected ? Palette.tileSelectedInk : Palette.text.opacity(0.88))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(selected ? AnyShapeStyle(Palette.tileSelected)
                                   : AnyShapeStyle(LinearGradient(colors: [Palette.cardHi, Palette.card], startPoint: .top, endPoint: .bottom)))
            )
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(selected ? .clear : Palette.stroke))
            .shadow(color: selected ? Color(hex: 0x8EC5F5).opacity(0.28) : .clear, radius: 12, y: 4)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// A level bar you can drag. Maps -24...+12 dB onto the track; filled sky
/// blue when the sound is let through, slate when it is turned down.
struct GainBar: View {
    let gain: Float
    let floor: Float
    var onCommit: (Float) -> Void
    @State private var dragging: Float?

    static let lo: Float = -24, hi: Float = 12

    private func fraction(_ g: Float) -> CGFloat { CGFloat((g - Self.lo) / (Self.hi - Self.lo)) }

    var body: some View {
        let value = dragging ?? gain
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.08))
                Capsule()
                    .fill(value >= -2 ? AnyShapeStyle(LinearGradient(colors: [Palette.sky.opacity(0.75), Palette.sky], startPoint: .leading, endPoint: .trailing))
                                     : AnyShapeStyle(Palette.quiet))
                    .frame(width: max(8, geo.size.width * fraction(value)))
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { g in
                        let f = Float(max(0, min(1, g.location.x / geo.size.width)))
                        dragging = max(floor, (Self.lo + f * (Self.hi - Self.lo)).rounded())
                    }
                    .onEnded { _ in
                        if let d = dragging { onCommit(d) }
                        dragging = nil
                    }
            )
        }
        .frame(height: 8)
        .animation(.smooth(duration: 0.25), value: gain)
        .accessibilityElement()
        .accessibilityValue(formatDB(value))
        .accessibilityAdjustableAction { dir in
            onCommit(max(floor, min(Self.hi, value + (dir == .increment ? 3 : -3))))
        }
    }
}

/// One row of "Selective Sound Control".
struct SoundControlRow: View {
    @Environment(AppModel.self) private var model
    let category: SoundCategory

    var body: some View {
        let d = model.decision(category)
        let hearing = model.isHearing(category)
        let letThrough = d.gainDB >= -2
        let floor: Float = model.profile.safetySet.contains(category) ? SafetyGuard.minGainDB : GainBar.lo

        HStack(spacing: 12) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: category.info.symbol)
                    .font(.system(size: 16))
                    .foregroundStyle(Palette.text.opacity(0.85))
                    .frame(width: 26)
                if hearing {
                    Circle().fill(Palette.sky).frame(width: 6, height: 6)
                        .offset(x: 3, y: -2)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 5) {
                    Text(category.label(in: model.mode, personLock: model.lockedName))
                        .font(Type_.regular(14.5))
                        .foregroundStyle(Palette.text)
                        .lineLimit(1)
                    if category.info.safetyCritical && model.profile.safetySet.contains(category) {
                        Image(systemName: "shield.fill").font(.system(size: 9)).foregroundStyle(Palette.safety.opacity(0.85))
                    }
                }
                if hearing {
                    Text("Hearing now").font(Type_.regular(10.5)).foregroundStyle(Palette.sky.opacity(0.9))
                }
            }
            Spacer(minLength: 6)
            Button { model.toggle(category) } label: {
                Image(systemName: letThrough ? "plus" : "minus")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(letThrough ? Palette.skyDeep : Palette.text2)
                    .frame(width: 34, height: 30)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(letThrough ? "Turn \(category.info.label) down" : "Let \(category.info.label) through")
            GainBar(gain: d.gainDB, floor: floor) { model.setGain(category, $0) }
                .frame(width: 84)
        }
        .padding(.vertical, 9)
        .animation(.smooth(duration: 0.3), value: hearing)
        .contextMenu {
            if model.profile.override(model.mode, category) != nil {
                Button("Reset to \(model.mode.title) default") { model.setGain(category, nil) }
            }
        }
    }
}

/// Frosted white chip, as on the brand's mode cards.
struct SoundChip: View {
    let symbol: String
    let label: String
    let plus: Bool

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: symbol).font(.system(size: 11))
            Text(label).font(Type_.medium(11)).lineLimit(1).minimumScaleFactor(0.8)
            Spacer(minLength: 4)
            Image(systemName: plus ? "plus" : "minus")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(plus ? Palette.skyDeep : Color(hex: 0x6B7482))
        }
        .foregroundStyle(Color(hex: 0x2A323D))
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(width: 150)
        .background(.white.opacity(0.9), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .shadow(color: .black.opacity(0.18), radius: 6, y: 2)
    }
}

/// Input vs output envelopes: where AURA removes energy, the blue sits inside
/// the grey. The clearest single picture of the product working.
struct WaveformView: View {
    let input: [Float]
    let output: [Float]

    var body: some View {
        Canvas { ctx, size in
            let mid = size.height / 2
            func draw(_ d: [Float], _ fill: Color, _ stroke: Color) {
                guard d.count > 1 else { return }
                let step = size.width / CGFloat(d.count - 1)
                var p = Path()
                for (i, v) in d.enumerated() {
                    let y = mid - CGFloat(min(v, 1)) * (mid - 2)
                    i == 0 ? p.move(to: CGPoint(x: 0, y: y)) : p.addLine(to: CGPoint(x: CGFloat(i) * step, y: y))
                }
                for (i, v) in d.enumerated().reversed() {
                    p.addLine(to: CGPoint(x: CGFloat(i) * step, y: mid + CGFloat(min(v, 1)) * (mid - 2)))
                }
                p.closeSubpath()
                ctx.fill(p, with: .color(fill))
                ctx.stroke(p, with: .color(stroke), lineWidth: 1)
            }
            draw(input, Palette.text2.opacity(0.14), Palette.text2.opacity(0.45))
            draw(output, Palette.sky.opacity(0.22), Palette.sky)
            ctx.fill(Path(CGRect(x: 0, y: mid - 0.25, width: size.width, height: 0.5)), with: .color(.white.opacity(0.08)))
        }
    }
}

struct Toast: View {
    let text: String
    var body: some View {
        Text(text)
            .font(Type_.medium(13))
            .foregroundStyle(Palette.text)
            .padding(.horizontal, 16).padding(.vertical, 11)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().stroke(Palette.stroke))
            .shadow(color: .black.opacity(0.3), radius: 12, y: 6)
    }
}

struct SectionHeader: View {
    let title: String
    var info: String?
    @State private var showInfo = false

    var body: some View {
        HStack(spacing: 6) {
            Text(title).font(Type_.medium(15)).foregroundStyle(Palette.text)
            if let info {
                Button { showInfo.toggle() } label: {
                    Image(systemName: "info.circle").font(.system(size: 13)).foregroundStyle(Palette.text2)
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showInfo) {
                    Text(info).font(Type_.regular(13)).padding().frame(width: 280).presentationCompactAdaptation(.popover)
                }
            }
            Spacer()
        }
    }
}

/// Safety override banner. Shown whenever the SafetyGuard is holding a sound
/// audible against the mode or the mixer.
struct SafetyBanner: View {
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "shield.lefthalf.filled").font(.system(size: 16))
            VStack(alignment: .leading, spacing: 1) {
                Text("Safety Override").font(Type_.semibold(14))
                Text("A warning sound is being kept audible.").font(Type_.regular(12)).opacity(0.8)
            }
            Spacer()
        }
        .foregroundStyle(Color(hex: 0xFFE1DC))
        .padding(14)
        .background(Palette.safety.opacity(0.16), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Palette.safety.opacity(0.4)))
        .accessibilityElement(children: .combine)
    }
}
