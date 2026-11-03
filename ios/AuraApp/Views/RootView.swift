import AuraCore
import SwiftUI

struct RootView: View {
    @Environment(AppModel.self) private var model
    @State private var tab: Tab = Tab.allCases.first { $0.rawValue.lowercased() == LaunchOptions.screen } ?? .home
    @State private var showDemo = LaunchOptions.screen == "demo"

    enum Tab: String, CaseIterable {
        case home = "Home", modes = "Modes", insights = "Insights", you = "You"
        var symbol: String {
            switch self {
            case .home: "house.fill"
            case .modes: "circle.grid.2x2"
            case .insights: "waveform.path.ecg"
            case .you: "person"
            }
        }
    }

    var body: some View {
        @Bindable var model = model
        ZStack(alignment: .bottom) {
            Screen {
                Group {
                    switch tab {
                    case .home: HomeView(openTab: { tab = $0 })
                    case .modes: ModesView()
                    case .insights: InsightsView()
                    case .you: YouView()
                    }
                }
                .safeAreaPadding(.bottom, 60)
            }
            TabBar(tab: $tab)
            if let t = model.toast {
                Toast(text: t).padding(.bottom, 96).transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.smooth(duration: 0.3), value: model.toast)
        .fullScreenCover(isPresented: .constant(!model.profile.onboarded || LaunchOptions.onboardingStep != nil)) {
            OnboardingView()
        }
        .sheet(isPresented: $showDemo) { DemoView() }
        .task { model.applyLaunchOptions() }
        .alert("AURA", isPresented: Binding(get: { model.alert != nil }, set: { if !$0 { model.alert = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.alert ?? "")
        }
    }
}

struct TabBar: View {
    @Binding var tab: RootView.Tab

    var body: some View {
        HStack {
            ForEach(RootView.Tab.allCases, id: \.self) { t in
                Button {
                    withAnimation(.smooth(duration: 0.25)) { tab = t }
                } label: {
                    VStack(spacing: 5) {
                        Image(systemName: t.symbol).font(.system(size: 18))
                        Text(t.rawValue).font(Type_.medium(10.5))
                    }
                    .foregroundStyle(tab == t ? Palette.text : Palette.text3)
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(tab == t ? .isSelected : [])
            }
        }
        .padding(.top, 12)
        .padding(.bottom, 2)
        .background(
            Rectangle().fill(Color(hex: 0x151A21))
                .overlay(alignment: .top) { Rectangle().fill(Palette.stroke).frame(height: 1) }
                .ignoresSafeArea()
        )
    }
}
