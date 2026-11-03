import Accelerate
import Foundation

let EPS: Float = 1e-10

/// Fixed-capacity history of recent values, oldest first on read.
/// Storage is allocated once; pushing never allocates.
struct History {
    private var storage: [Float]
    private var head = 0
    private(set) var count = 0
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        storage = [Float](repeating: 0, count: capacity)
    }

    mutating func push(_ v: Float) {
        storage[head] = v
        head = (head + 1) % capacity
        count = min(count + 1, capacity)
    }

    mutating func clear() { head = 0; count = 0 }

    /// Value `i` in chronological order (0 = oldest).
    func at(_ i: Int) -> Float {
        storage[(head - count + i + capacity) % capacity]
    }

    func mean() -> Float {
        guard count > 0 else { return 0 }
        var s: Float = 0
        for i in 0..<count { s += at(i) }
        return s / Float(count)
    }

    /// Population variance (NumPy's default ddof=0).
    func variance() -> Float {
        guard count > 0 else { return 0 }
        let m = mean()
        var s: Float = 0
        for i in 0..<count { let d = at(i) - m; s += d * d }
        return s / Float(count)
    }
}

/// Symmetric Hann window, identical to `np.hanning(n)`.
func hanning(_ n: Int) -> [Float] {
    guard n > 1 else { return [1] }
    return (0..<n).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(n - 1)) }
}

/// Least-squares slope of y against x (`np.polyfit(x, y, 1)[0]`).
func linearSlope(_ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>, _ n: Int) -> Float {
    guard n > 1 else { return 0 }
    var sx: Double = 0, sy: Double = 0, sxx: Double = 0, sxy: Double = 0
    for i in 0..<n {
        let a = Double(x[i]), b = Double(y[i])
        sx += a; sy += b; sxx += a * a; sxy += a * b
    }
    let dn = Double(n)
    let den = dn * sxx - sx * sx
    guard abs(den) > 1e-12 else { return 0 }
    return Float((dn * sxy - sx * sy) / den)
}

@inline(__always) func clamp(_ v: Float, _ lo: Float, _ hi: Float) -> Float { min(max(v, lo), hi) }

@inline(__always) func toDB(_ rms: Float) -> Float { 20 * log10f(max(rms, 1e-9)) }

/// 0 below `lo`, 1 above `hi`, linear between.
@inline(__always) func ramp(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
    if hi <= lo { return x >= hi ? 1 : 0 }
    return clamp((x - lo) / (hi - lo), 0, 1)
}

@inline(__always) func invRamp(_ x: Float, _ lo: Float, _ hi: Float) -> Float { 1 - ramp(x, lo, hi) }

/// Trapezoid: 1 on [pLo, pHi], tapering to 0 at `lo` and `hi`.
@inline(__always) func window(_ x: Float, _ lo: Float, _ pLo: Float, _ pHi: Float, _ hi: Float) -> Float {
    if x < pLo { return ramp(x, lo, pLo) }
    if x > pHi { return invRamp(x, pHi, hi) }
    return 1
}

/// Notched moving-average local floor, as used by the tonality measure and the
/// transient selector. `out[i]` is the mean of `mag` over a window around `i`
/// that excludes `notch` bins either side of `i` itself.
func notchedFloor(_ mag: UnsafePointer<Float>, count n: Int, halfWidth c: Int, notch: Int,
                  into out: UnsafeMutablePointer<Float>) {
    // Kernel weight: ones over [-c, c] except [-notch, notch]; normalised.
    let kept = Float((2 * c + 1) - (2 * min(notch, c) + 1))
    guard kept > 0 else {
        for i in 0..<n { out[i] = 0 }
        return
    }
    let inv = 1 / kept
    for i in 0..<n {
        var s: Float = 0
        let lo = max(0, i - c), hi = min(n - 1, i + c)
        var j = lo
        while j <= hi {
            let d = j - i
            if d < -notch || d > notch { s += mag[j] }
            j += 1
        }
        out[i] = s * inv
    }
}
