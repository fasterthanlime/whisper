import Cocoa
import InputMethodKit

/// Custom NSApplication subclass that sets up the AppDelegate manually.
/// InputMethodKit requires this pattern instead of @main / @NSApplicationMain.
class HarkInputApplication: NSApplication {
    private let appDelegate = HarkInputAppDelegate()

    override init() {
        super.init()
        self.delegate = appDelegate
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not supported")
    }
}

// Launch the application
let app = HarkInputApplication.shared
NSApp.run()
