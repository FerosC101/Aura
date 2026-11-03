import Foundation

/// Debug-only launch arguments for screenshots and UI review, e.g.
///
///     xcrun simctl launch booted com.aura.selectiveawareness \
///         -AuraScreen insights -AuraDemo psc_demo -AuraName Vince
///
/// Compiled out of release builds entirely.
enum LaunchOptions {
    #if DEBUG
    static var screen: String? { UserDefaults.standard.string(forKey: "AuraScreen") }
    static var demo: String? { UserDefaults.standard.string(forKey: "AuraDemo") }
    static var name: String? { UserDefaults.standard.string(forKey: "AuraName") }
    static var mode: String? { UserDefaults.standard.string(forKey: "AuraMode") }
    /// Opens a sheet straight away, e.g. `-AuraSheet voicefocus`.
    static var sheet: String? { UserDefaults.standard.string(forKey: "AuraSheet") }
    /// Comma-separated names to enrol from bundled demo speech, so screenshots
    /// of Voice Focus show the real list rather than the empty state. Goes
    /// through the ordinary enrolment path -- no hand-made voiceprints.
    static var voices: [String] {
        (UserDefaults.standard.string(forKey: "AuraVoices") ?? "")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
    static var onboardingStep: Int? {
        UserDefaults.standard.object(forKey: "AuraOnboarding") == nil ? nil : UserDefaults.standard.integer(forKey: "AuraOnboarding")
    }
    #else
    static let screen: String? = nil
    static let demo: String? = nil
    static let name: String? = nil
    static let mode: String? = nil
    static let sheet: String? = nil
    static let voices: [String] = []
    static let onboardingStep: Int? = nil
    #endif
}
