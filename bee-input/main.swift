import Cocoa
import InputMethodKit

class BeeInputApplication: NSApplication {
    let appDelegate = BeeInputAppDelegate()

    override init() {
        super.init()
        self.delegate = appDelegate
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError()
    }
}

let app = BeeInputApplication.shared
NSApp.run()
