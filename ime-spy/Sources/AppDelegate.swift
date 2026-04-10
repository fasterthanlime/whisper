import AppKit

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(
            contentRect: NSRect(x: 200, y: 200, width: 600, height: 200),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "IME Spy"

        let spyView = SpyTextView(frame: window.contentView!.bounds)
        spyView.autoresizingMask = [.width, .height]
        window.contentView!.addSubview(spyView)

        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(spyView)

        NSApp.activate(ignoringOtherApps: true)

        spy("IME Spy ready. Focus this window and use your IME.")
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

func spy(_ msg: String) {
    let df = DateFormatter()
    df.dateFormat = "HH:mm:ss.SSS"
    print("\(df.string(from: Date())) SPY: \(msg)")
}
