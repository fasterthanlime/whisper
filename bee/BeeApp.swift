import SwiftUI

@main
struct BeeApp: App {
    @State private var appState: AppState

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
        } label: {
            // TODO: use bee-bw.png as menu bar icon
            Image(systemName: "waveform.circle")
        }
    }
}

struct MenuBarView: View {
    let appState: AppState

    var body: some View {
        // h[impl menubar.status]
        Text(statusText)
            .font(.headline)

        Divider()

        // h[impl menubar.history]
        Section("Recent") {
            Text("No recent transcriptions")
                .foregroundStyle(.secondary)
        }

        Divider()

        // h[impl menubar.model]
        // h[impl menubar.input-device]
        // h[impl menubar.warm-toggle]
        // h[impl menubar.run-on-startup]
        // h[impl menubar.pause-media]
        Section("Settings") {
            Text("TODO: model, device, toggles")
                .foregroundStyle(.secondary)
        }

        Divider()

        // h[impl menubar.quit]
        // h[impl ime.safety.restore-on-quit]
        Button("Quit Bee") {
            BeeInputClient.restoreInputSourceIfNeeded()
            NSApplication.shared.terminate(nil)
        }
    }

    private var statusText: String {
        switch appState.uiState {
        case .idle: "Bee Ready"
        case .pending: "Bee Starting..."
        case .pushToTalk: "Bee Recording"
        case .locked: "Bee Recording (Locked)"
        case .lockedOptionHeld: "Bee Recording (Locked)"
        }
    }
}
