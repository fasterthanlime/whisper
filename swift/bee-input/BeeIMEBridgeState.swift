import AppKit
import Foundation

/// Tracks which controller the system gave us and what session we're serving.
final class BeeIMEBridgeState: NSObject {
    static let shared = BeeIMEBridgeState()

    struct ActiveClient {
        weak var controller: BeeInputController?
        let pid: pid_t?
        let clientID: String?
    }

    enum State {
        /// No controller — IME is not active for any text field.
        case idle
        /// activateServer fired, we have a live controller, but no session yet.
        case activated(ActiveClient)
        /// Controller is live and serving a dictation session.
        case serving(ActiveClient, sessionID: UUID, pendingText: String?)
    }

    private(set) var state: State = .idle

    // MARK: - Queries

    var isDictating: Bool {
        if case .serving = state { return true }
        return false
    }

    var activeSessionID: UUID? {
        if case .serving(_, let sessionID, _) = state { return sessionID }
        return nil
    }

    var activeController: BeeInputController? {
        switch state {
        case .idle: return nil
        case .activated(let client): return client.controller
        case .serving(let client, _, _): return client.controller
        }
    }

    // MARK: - State transitions

    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?) {
        let client = ActiveClient(controller: controller, pid: pid, clientID: clientID)
        state = .activated(client)
        beeInputLog("state → activated pid=\(pid.map(String.init) ?? "nil")")
    }

    func deactivate(_ controller: BeeInputController) {
        // Only deactivate if this is the current controller
        guard activeController === controller else { return }
        state = .idle
    }

    func attachSession(sessionID: UUID) {
        let client: ActiveClient
        switch state {
        case .activated(let c):
            client = c
        case .serving(let c, _, _):
            client = c
        case .idle:
            beeInputLog("attachSession: idle, ignoring session=\(sessionID.uuidString.prefix(8))")
            return
        }
        state = .serving(client, sessionID: sessionID, pendingText: nil)
        beeInputLog("state → serving session=\(sessionID.uuidString.prefix(8))")
    }

    func clearSessionIfMatching(sessionID: UUID) {
        guard case .serving(let client, let currentID, _) = state, currentID == sessionID else { return }
        beeInputLog("clearSession: session=\(sessionID.uuidString.prefix(8))")
        state = .activated(client)
    }

    func endSession() {
        guard case .serving(let client, _, _) = state else { return }
        state = .activated(client)
    }

    // MARK: - Text routing (called from XPC callbacks, dispatches to main)

    func flushPending() {
        guard case .serving(let client, let sessionID, let text?) = state,
              let ctrl = client.controller else { return }
        beeInputLog("flushPending: delivering \(text.prefix(40).debugDescription)")
        state = .serving(client, sessionID: sessionID, pendingText: nil)
        ctrl.handleSetMarkedText(text)
    }

    func setMarkedText(_ text: String, sessionID: UUID) {
        DispatchQueue.main.async {
            guard case .serving(let client, let currentID, _) = self.state, currentID == sessionID else {
                beeInputLog("setMarkedText: stale session=\(sessionID.uuidString.prefix(8)), dropping")
                return
            }
            if let ctrl = client.controller {
                ctrl.handleSetMarkedText(text)
            } else {
                beeInputLog("setMarkedText: no controller, queuing")
                self.state = .serving(client, sessionID: sessionID, pendingText: text)
            }
        }
    }

    func commitText(_ text: String, submit: Bool, sessionID: UUID) {
        DispatchQueue.main.async {
            guard case .serving(let client, let currentID, _) = self.state, currentID == sessionID else {
                beeInputLog("commitText: stale session=\(sessionID.uuidString.prefix(8)), dropping")
                return
            }
            self.state = .activated(client)
            client.controller?.handleCommitText(text, submit: submit)
        }
    }

    func cancelInput(sessionID: UUID) {
        DispatchQueue.main.async {
            guard case .serving(let client, let currentID, _) = self.state, currentID == sessionID else {
                beeInputLog("cancelInput: stale session=\(sessionID.uuidString.prefix(8)), dropping")
                return
            }
            self.state = .activated(client)
            client.controller?.handleCancelInput()
        }
    }

    func stopDictating(sessionID: UUID) {
        DispatchQueue.main.async {
            guard case .serving(let client, let currentID, _) = self.state, currentID == sessionID else {
                beeInputLog("stopDictating: stale session=\(sessionID.uuidString.prefix(8)), dropping")
                return
            }
            self.state = .activated(client)
        }
    }
}
