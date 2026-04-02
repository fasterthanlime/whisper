import SwiftUI

struct DebugOverlay: View {
    let appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("bee debug")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)

            Divider()

            row("ui", uiStateLabel)
            row("model", modelLabel)

            if let session = appState.hotkeyState.session {
                Divider()
                Text("active session")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.orange)
                SessionDebugView(sessionDiag: session.diag)
            }

            if let diag = appState.lastSessionDiag {
                Divider()
                Text("last session")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.green)
                DiagnosticsView(diag: diag, transcriptionService: appState.transcriptionService)
            }
        }
        .padding(8)
        .frame(width: 300, alignment: .leading)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.white.opacity(0.1), lineWidth: 0.5)
        }
    }

    private var uiStateLabel: String {
        let hotkey: String = switch appState.hotkeyState {
        case .idle: "Idle"
        case .held: "Held"
        case .released: "Released"
        case .pushToTalk: "PushToTalk"
        case .locked: "Locked"
        case .lockedOptionHeld: "LockedOptionHeld"
        }
        let ime: String = switch appState.imeSessionState {
        case .inactive: "Inactive"
        case .activating: "Activating"
        case .active: "Active"
        case .parked: "Parked"
        }
        return "\(hotkey) | IME: \(ime)"
    }

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: "not loaded"
        case .downloading(let p): "downloading \(Int(p * 100))%"
        case .loading: "loading..."
        case .loaded: "ready"
        case .error(let e): "error: \(e.prefix(30))"
        }
    }

    private func row(_ label: some StringProtocol, _ value: some StringProtocol) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .trailing)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

struct SessionDebugView: View {
    let sessionDiag: SessionDiag

    @State private var snapshot: SessionDiag.Snapshot?

    var body: some View {
        Group {
            if let snapshot {
                DiagnosticsView(diag: snapshot)
            } else {
                Text("loading...")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
        .task(id: UUID()) {
            while !Task.isCancelled {
                // No actor hop — just reads a lock-protected struct
                snapshot = sessionDiag.snapshot
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }
}

struct DiagnosticsView: View {
    let diag: SessionDiag.Snapshot
    var transcriptionService: TranscriptionService? = nil

    @State private var batchResult: String?
    @State private var isBatchRunning = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            row("ending", diag.ending.isEmpty ? "—" : diag.ending)
            row("rec dur", "\(diag.recordingDurationMs) ms")

            Divider().padding(.vertical, 1)

            row("feeds", "\(diag.feeds) (last \(diag.lastFeedUs / 1000)ms, total \(diag.totalFeedUs / 1000)ms)")
            row("captured", "\(diag.capturedSamples) @16k")
            row("fed", "\(diag.fedSamples) @16k")
            row("total", "\(diag.totalSamples) @16k (\(diag.totalAudioDurationMs) ms)")

            if diag.drainBuffers > 0 {
                row("drain", "\(diag.drainBuffers) bufs (\(diag.drainSamples) @16k)")
            }

            Divider().padding(.vertical, 1)

            row("finalize", "\(diag.finalizeUs / 1000) ms")

            if !diag.finalText.isEmpty {
                Text(diag.finalText.prefix(100))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.cyan)
                    .lineLimit(3)
                    .padding(.leading, 66)
            }

            if !diag.audioWavPath.isEmpty {
                Divider().padding(.vertical, 1)

                HStack(spacing: 8) {
                    Button("Play") {
                        NSSound(contentsOfFile: diag.audioWavPath, byReference: true)?.play()
                    }
                    .font(.system(size: 10, weight: .semibold))

                    if let transcriptionService {
                        Button(isBatchRunning ? "Running..." : "Re-transcribe (batch)") {
                            guard !isBatchRunning else { return }
                            isBatchRunning = true
                            batchResult = nil
                            Task.detached {
                                let samples = loadWavSamples(path: diag.audioWavPath)
                                let result = transcriptionService.transcribeSamples(samples)
                                await MainActor.run {
                                    batchResult = result ?? "(empty)"
                                    isBatchRunning = false
                                }
                            }
                        }
                        .font(.system(size: 10, weight: .semibold))
                        .disabled(isBatchRunning)
                    }

                    Button("Reveal") {
                        NSWorkspace.shared.selectFile(diag.audioWavPath, inFileViewerRootedAtPath: "")
                    }
                    .font(.system(size: 10, weight: .semibold))
                }
                .padding(.leading, 66)

                if let batchResult {
                    Text("batch: \(batchResult.prefix(120))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.yellow)
                        .lineLimit(3)
                        .padding(.leading, 66)
                }
            }
        }
    }

    private func row(_ label: some StringProtocol, _ value: some StringProtocol) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .trailing)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

/// Load a 16-bit PCM WAV as float32 samples.
private func loadWavSamples(path: String) -> [Float] {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return [] }
    guard data.count > 44 else { return [] }
    let pcmData = data.dropFirst(44)
    var samples: [Float] = []
    samples.reserveCapacity(pcmData.count / 2)
    for i in stride(from: 0, to: pcmData.count - 1, by: 2) {
        let lo = UInt16(pcmData[pcmData.startIndex + i])
        let hi = UInt16(pcmData[pcmData.startIndex + i + 1])
        let int16 = Int16(bitPattern: lo | (hi << 8))
        samples.append(Float(int16) / 32767.0)
    }
    return samples
}
