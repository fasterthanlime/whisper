import Foundation

struct TranscriptionHistoryItem: Identifiable, Sendable {
    let id: UUID
    let text: String
    let timestamp: Date

    init(text: String) {
        self.id = UUID()
        self.text = text
        self.timestamp = Date()
    }

    var displayText: String {
        let truncated = text.prefix(60)
        let firstLine = truncated.prefix(while: { $0 != "\n" })
        if firstLine.count < text.count {
            return "..." + firstLine + "..."
        }
        return String(firstLine)
    }
}
