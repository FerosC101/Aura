import SwiftUI

/// A drawn mountain lake at first light -- the brand's photography, rendered
/// procedurally so it ships with no image assets and can shift mood per mode.
struct Landscape: View {
    enum Mood { case dawn, dusk, forest, golden, mist }
    var mood: Mood = .dawn
    var showLake = true
    var sunGlow = true

    private var sky: [Color] {
        switch mood {
        case .dawn: [Color(hex: 0x3B4A61), Color(hex: 0x7F93AE), Color(hex: 0xC9D3DE), Color(hex: 0xEAD9C6)]
        case .dusk: [Color(hex: 0x252C3A), Color(hex: 0x4A5873), Color(hex: 0x9C8FA0), Color(hex: 0xE3B79A)]
        case .forest: [Color(hex: 0x3A4E57), Color(hex: 0x7E9A9E), Color(hex: 0xC7D5CF), Color(hex: 0xE8E2CC)]
        case .golden: [Color(hex: 0x3E3A45), Color(hex: 0x8A7A7C), Color(hex: 0xD8B794), Color(hex: 0xF3D2A6)]
        case .mist: [Color(hex: 0x39414E), Color(hex: 0x76849A), Color(hex: 0xB8C2CF), Color(hex: 0xDADFE5)]
        }
    }

    private var ridges: [Color] {
        switch mood {
        case .forest: [Color(hex: 0x6F868A), Color(hex: 0x40575A), Color(hex: 0x223234)]
        case .golden: [Color(hex: 0x8C7C80), Color(hex: 0x514750), Color(hex: 0x2A2530)]
        default: [Color(hex: 0x7B8AA0), Color(hex: 0x4B586B), Color(hex: 0x252E3A)]
        }
    }

    var body: some View {
        Canvas { ctx, size in
            let w = size.width, h = size.height
            let horizon = h * (showLake ? 0.62 : 0.78)

            ctx.fill(Path(CGRect(origin: .zero, size: size)),
                     with: .linearGradient(Gradient(colors: sky), startPoint: .zero, endPoint: CGPoint(x: 0, y: horizon)))

            if sunGlow {
                let c = CGPoint(x: w * 0.56, y: horizon)
                ctx.fill(Path(CGRect(x: 0, y: 0, width: w, height: horizon)),
                         with: .radialGradient(Gradient(colors: [Color(hex: 0xFFE8CC, opacity: 0.55), .clear]),
                                               center: c, startRadius: 0, endRadius: w * 0.5))
            }

            func ridge(_ base: CGFloat, _ amp: CGFloat, _ seed: CGFloat) -> Path {
                var p = Path()
                p.move(to: CGPoint(x: 0, y: horizon + 1))
                for i in 0...120 {
                    let t = CGFloat(i) / 120
                    // Asymmetric peaks: sharper rise than fall, like real ridgelines.
                    let n = 0.55 * sin(t * 7.3 + seed) + 0.28 * sin(t * 17.1 + seed * 1.7)
                        + 0.12 * sin(t * 41 + seed * 2.9) + 0.05 * sin(t * 97 + seed)
                    p.addLine(to: CGPoint(x: w * t, y: horizon - base - amp * max(n, -0.35)))
                }
                p.addLine(to: CGPoint(x: w, y: horizon + 1))
                p.closeSubpath()
                return p
            }

            let layers: [(CGFloat, CGFloat, CGFloat)] = [(h * 0.20, h * 0.10, 1.2), (h * 0.12, h * 0.07, 3.9), (h * 0.05, h * 0.05, 6.4)]
            for (i, l) in layers.enumerated() {
                let p = ridge(l.0, l.1, l.2)
                ctx.fill(p, with: .color(ridges[i].opacity(i == 0 ? 0.55 : 0.9)))
            }

            if showLake {
                let lake = CGRect(x: 0, y: horizon, width: w, height: h - horizon)
                ctx.fill(Path(lake), with: .linearGradient(
                    Gradient(colors: [sky[2].opacity(0.85), sky[1].opacity(0.8), Color(hex: 0x1A212B)]),
                    startPoint: CGPoint(x: 0, y: horizon), endPoint: CGPoint(x: 0, y: h)))
                // Reflection: the ridges mirrored and softened.
                var mirror = ctx
                mirror.opacity = 0.35
                mirror.translateBy(x: 0, y: horizon * 2)
                mirror.scaleBy(x: 1, y: -1)
                mirror.clip(to: Path(CGRect(x: 0, y: horizon - (h - horizon), width: w, height: h - horizon)))
                for (i, l) in layers.enumerated().reversed() where i > 0 {
                    mirror.fill(ridge(l.0, l.1, l.2), with: .color(ridges[i]))
                }
                // Shimmer lines on the water.
                for k in 0..<9 {
                    let y = horizon + (h - horizon) * (0.12 + 0.09 * CGFloat(k))
                    let x0 = w * (0.3 + 0.05 * sin(CGFloat(k) * 2.1)), len = w * (0.18 + 0.06 * CGFloat(k % 3))
                    ctx.fill(Path(CGRect(x: x0, y: y, width: len, height: 0.8)), with: .color(.white.opacity(0.10)))
                }
            }
        }
    }
}
