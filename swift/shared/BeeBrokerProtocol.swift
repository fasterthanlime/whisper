import Foundation

@objc
protocol BeeBrokerXPC {
    func appHello(_ appInstanceID: String, withReply reply: @escaping (Bool) -> Void)
    func imeHello(_ imeInstanceID: String, withReply reply: @escaping (Bool) -> Void)

    func waitForIME(appInstanceID: String, withReply reply: @escaping (Bool) -> Void)

    func prepareSession(
        _ sessionID: String,
        activationID: String,
        targetPID: Int32,
        appInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    )
    /// Returns (found, sessionID, shouldStayActive).
    /// shouldStayActive is true if the IME should remain selected even
    /// when no session was claimed (e.g. session recently ended).
    func claimPreparedSession(
        imeInstanceID: String,
        withReply reply: @escaping (Bool, String, Bool, Int32) -> Void
    )
    func clearSession(_ sessionID: String, appInstanceID: String, withReply reply: @escaping () -> Void)

    func setMarkedText(
        _ sessionID: String,
        text: String,
        appInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    )
    func commitText(
        _ sessionID: String,
        text: String,
        submit: Bool,
        appInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    )
    func cancelInput(_ sessionID: String, appInstanceID: String, withReply reply: @escaping (Bool) -> Void)
    func stopDictating(_ sessionID: String, appInstanceID: String, withReply reply: @escaping (Bool) -> Void)

    func imeAttach(
        _ sessionID: String,
        imeInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    )
    func imeSubmit(_ sessionID: String, imeInstanceID: String, withReply reply: @escaping () -> Void)
    func imeCancel(_ sessionID: String, imeInstanceID: String, withReply reply: @escaping () -> Void)
    func imeUserTyped(
        _ sessionID: String,
        keyCode: Int32,
        characters: String,
        imeInstanceID: String,
        withReply reply: @escaping () -> Void
    )
    func imeContextLost(
        _ sessionID: String,
        hadMarkedText: Bool,
        imeInstanceID: String,
        withReply reply: @escaping () -> Void
    )
    func imeActivationRevoked(
        imeInstanceID: String,
        withReply reply: @escaping () -> Void
    )
}

@objc
protocol BeeBrokerPeerXPC {
    // Broker -> IME: new session prepared, try claiming without activateServer.
    func handleNewPreparedSession(_ sessionID: String, targetPID: Int32)

    // App -> IME forwarded commands.
    func handleClearSession(_ sessionID: String)
    func handleSetMarkedText(_ sessionID: String, text: String)
    func handleCommitText(_ sessionID: String, text: String, submit: Bool)
    func handleCancelInput(_ sessionID: String)
    func handleStopDictating(_ sessionID: String)

    // IME -> App forwarded events.
    func handleIMESessionStarted(_ sessionID: String)
    func handleIMESubmit(_ sessionID: String)
    func handleIMECancel(_ sessionID: String)
    func handleIMEUserTyped(_ sessionID: String, keyCode: Int32, characters: String)
    func handleIMEContextLost(_ sessionID: String, hadMarkedText: Bool)
    func handleIMEActivationRevoked()
}
