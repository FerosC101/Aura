// Renders the AURA app icon: the chevron "A" of the ΛURΛ wordmark over a dawn
// sky -- dark slate above, a warm band of sunrise light at the horizon, a
// faint mountain line. Run: swift tools/render_icon.swift <out.png>
import AppKit
import CoreGraphics

let size = 1024
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "AppIcon.png"
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                    space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let S = CGFloat(size)

func rgb(_ h: UInt32, _ a: CGFloat = 1) -> CGColor {
    CGColor(red: CGFloat((h >> 16) & 0xff) / 255, green: CGFloat((h >> 8) & 0xff) / 255, blue: CGFloat(h & 0xff) / 255, alpha: a)
}

// Sky: slate at the top, easing into misty blue, warm at the horizon.
let sky = CGGradient(colorsSpace: cs, colors: [rgb(0x131820), rgb(0x2A3444), rgb(0x6E7F96), rgb(0xE9C7A6)] as CFArray,
                     locations: [0, 0.45, 0.78, 1])!
ctx.drawLinearGradient(sky, start: CGPoint(x: 0, y: S), end: CGPoint(x: 0, y: S * 0.18), options: [.drawsAfterEndLocation])

// Sun glow at the horizon.
let glow = CGGradient(colorsSpace: cs, colors: [rgb(0xFFE2C2, 0.85), rgb(0xFFD2A8, 0)] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: S * 0.5, y: S * 0.2), startRadius: 0,
                       endCenter: CGPoint(x: S * 0.5, y: S * 0.2), endRadius: S * 0.42, options: [])

// Mountain silhouettes.
func ridge(base: CGFloat, amp: CGFloat, seed: CGFloat, color: CGColor) {
    let p = CGMutablePath()
    p.move(to: CGPoint(x: 0, y: 0))
    for i in 0...200 {
        let x = S * CGFloat(i) / 200
        let t = CGFloat(i) / 200
        let y = base + amp * (0.55 * sin(t * 9 + seed) + 0.3 * sin(t * 23 + seed * 2) + 0.15 * sin(t * 51 + seed * 3))
        p.addLine(to: CGPoint(x: x, y: y))
    }
    p.addLine(to: CGPoint(x: S, y: 0))
    p.closeSubpath()
    ctx.addPath(p)
    ctx.setFillColor(color)
    ctx.fillPath()
}
ridge(base: S * 0.23, amp: S * 0.035, seed: 1.3, color: rgb(0x4B586B, 0.9))
ridge(base: S * 0.17, amp: S * 0.028, seed: 4.1, color: rgb(0x2B3441))
ridge(base: S * 0.10, amp: S * 0.02, seed: 7.7, color: rgb(0x161B23))

// The chevron A: two thin rounded strokes with a soft glow.
let apex = CGPoint(x: S * 0.5, y: S * 0.72)
let left = CGPoint(x: S * 0.30, y: S * 0.36)
let right = CGPoint(x: S * 0.70, y: S * 0.36)
ctx.setLineCap(.round)
ctx.setLineJoin(.round)
for (w, a) in [(CGFloat(58), CGFloat(0.06)), (38, 0.10), (24, 0.18)] {
    ctx.setStrokeColor(rgb(0xDDEEFF, a)); ctx.setLineWidth(w)
    ctx.move(to: left); ctx.addLine(to: apex); ctx.addLine(to: right); ctx.strokePath()
}
ctx.setStrokeColor(rgb(0xFFFFFF)); ctx.setLineWidth(16)
ctx.move(to: left); ctx.addLine(to: apex); ctx.addLine(to: right); ctx.strokePath()

let img = ctx.makeImage()!
let rep = NSBitmapImageRep(cgImage: img)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
