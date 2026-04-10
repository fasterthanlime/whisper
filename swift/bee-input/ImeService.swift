import Foundation
import VoxRuntime

// MARK: - IME-side handler (handles calls FROM app)

final class ImeImpl: ImeHandler, @unchecked Sendable {
    func setMarkedText(text: String) async throws -> Bool {
        beeInputLog("VOXIPC: setMarkedText \(text)")
        await MainActor.run { Bridge.shared.setMarkedText(text) }
        return true
    }

    func setPhase(phase: ImePhase) async throws -> Bool {
        beeInputLog("VOXIPC: setPhase \(phase)")
        await MainActor.run { Bridge.shared.setPhase(phase) }
        return true
    }

    func commitText(text: String) async throws -> Bool {
        beeInputLog("VOXIPC: commitText \(text)")
        await MainActor.run { Bridge.shared.commitText(text) }
        return true
    }

    func stopDictating() async throws -> Bool {
        beeInputLog("VOXIPC: stopDictating")
        await MainActor.run { Bridge.shared.commitText("") }
        return true
    }
}
