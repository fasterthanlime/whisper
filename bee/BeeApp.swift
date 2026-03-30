import SwiftUI

@main
struct BeeApp: App {
    @State private var appState: AppState
    @State private var hotkeyMonitor = HotkeyMonitor()

    init() {
        let audioEngine = AudioEngine()
        let transcriptionService = TranscriptionService()
        let inputClient = BeeInputClient()
        _appState = State(initialValue: AppState(
            audioEngine: audioEngine,
            transcriptionService: transcriptionService,
            inputClient: inputClient
        ))
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(appState: appState)
                .onAppear {
                    hotkeyMonitor.appState = appState
                    hotkeyMonitor.start()
                    BeeInputClient.ensureIMERegistered()
                    appState.loadModelAtStartup()
                }
        } label: {
            Image("MenuBarIcon")
        }
    }
}
