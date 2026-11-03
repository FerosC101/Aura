// swift-tools-version: 6.0
//
// AuraCore — the on-device AURA Engine.
//
// A Swift port of the Python reference engine in /src/aura. It has no UI and no
// AVFoundation dependency: it takes blocks of samples and returns processed
// blocks, so it can be unit-tested on macOS with `swift test` and embedded in
// the iOS app unchanged.

import PackageDescription

let package = Package(
    name: "AuraCore",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "AuraCore", targets: ["AuraCore"]),
    ],
    targets: [
        .target(
            name: "AuraCore",
            // Optimised in every configuration. This is real-time DSP: an
            // unoptimised build runs ~80x slower (14 ms per 20 ms hop versus
            // 0.18 ms) and would drop audio when run from Xcode in Debug.
            swiftSettings: [.unsafeFlags(["-O"])]
        ),
        .testTarget(
            name: "AuraCoreTests",
            dependencies: ["AuraCore"],
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v5]
)
