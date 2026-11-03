import Foundation
import Synchronization

/// Single-producer / single-consumer float ring. Lock-free: the audio render
/// thread must never wait on anyone.
public final class SPSCRing: @unchecked Sendable {
    public let capacity: Int
    private let buf: UnsafeMutablePointer<Float>
    private let written = Atomic<Int>(0)
    private let read = Atomic<Int>(0)

    public init(capacity: Int) {
        self.capacity = capacity
        buf = .allocate(capacity: capacity)
        buf.initialize(repeating: 0, count: capacity)
    }

    deinit { buf.deallocate() }

    public var available: Int { written.load(ordering: .acquiring) - read.load(ordering: .acquiring) }

    /// Writes all of `n` samples or none; returns false (dropping them) when
    /// full. A backlog is latency, not safety.
    @discardableResult
    public func write(_ src: UnsafePointer<Float>, _ n: Int) -> Bool {
        let w = written.load(ordering: .relaxed)
        let r = read.load(ordering: .acquiring)
        guard capacity - (w - r) >= n else { return false }
        for i in 0..<n { buf[(w + i) % capacity] = src[i] }
        written.store(w + n, ordering: .releasing)
        return true
    }

    public func writeZeros(_ n: Int) {
        let w = written.load(ordering: .relaxed)
        for i in 0..<n { buf[(w + i) % capacity] = 0 }
        written.store(w + n, ordering: .releasing)
    }

    @discardableResult
    public func read(into dst: UnsafeMutablePointer<Float>, _ n: Int) -> Int {
        let r = read.load(ordering: .relaxed)
        let w = written.load(ordering: .acquiring)
        let k = min(n, w - r)
        for i in 0..<k { dst[i] = buf[(r + i) % capacity] }
        read.store(r + k, ordering: .releasing)
        return k
    }

    public func clear() {
        read.store(written.load(ordering: .acquiring), ordering: .releasing)
    }
}
