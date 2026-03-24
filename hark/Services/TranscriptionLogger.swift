import Foundation

/// Logs every transcription to a JSONL file for training data collection.
///
/// Each line is a JSON object with:
///   - text: the raw ASR output
///   - timestamp: ISO 8601
///   - app: bundle ID of the frontmost app
///   - model: which ASR model was used
actor TranscriptionLogger {
    static let shared = TranscriptionLogger()

    private let logURL: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("hark", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("transcription_log.jsonl")
    }()

    func log(text: String, app: String?) {
        guard !text.isEmpty else { return }

        let entry: [String: Any] = [
            "text": text,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "app": app ?? "unknown",
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: entry),
              let line = String(data: data, encoding: .utf8) else {
            return
        }

        if let handle = try? FileHandle(forWritingTo: logURL) {
            handle.seekToEndOfFile()
            handle.write(Data((line + "\n").utf8))
            handle.closeFile()
        } else {
            try? Data((line + "\n").utf8).write(to: logURL)
        }
    }
}
