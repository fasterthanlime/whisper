import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "CorrectionService")

final class CorrectionService: @unchecked Sendable {
    private let engine: BeeEngine
    private let lock = NSLock()
    private var loaded = false

    struct Output: Sendable {
        let sessionId: String
        let originalText: String
        let bestText: String
        let edits: [CorrectionEdit]
    }

    enum LoadError: LocalizedError {
        case failed(String)
        var errorDescription: String? {
            switch self {
            case .failed(let msg): "Correction engine load failed: \(msg)"
            }
        }
    }

    init(engine: BeeEngine) {
        self.engine = engine
    }

    // MARK: - Lifecycle

    func load(
        datasetDir: String, eventsPath: String?,
        gateThreshold: Float = 0.5, rankerThreshold: Float = 0.2
    ) async throws {
        try await engine.correctLoad(
            datasetDir: datasetDir,
            eventsPath: eventsPath ?? "",
            gateThreshold: gateThreshold,
            rankerThreshold: rankerThreshold
        )
        lock.withLock { loaded = true }
        logger.info("Correction engine loaded")
    }

    func unload() {
        lock.withLock { loaded = false }
    }

    // MARK: - Processing

    func process(text: String, appId: String?) async -> Output? {
        do {
            let result = try await engine.correctProcess(text: text, appId: appId ?? "")
            if result.edits.isEmpty {
                return nil
            }

            logger.info("Corrected \(result.edits.count) span(s): \"\(text.prefix(40))\" → \"\(result.bestText.prefix(40))\"")

            return Output(
                sessionId: result.sessionId,
                originalText: text,
                bestText: result.bestText,
                edits: result.edits
            )
        } catch {
            logger.error("process error: \(error)")
            return nil
        }
    }

    // MARK: - Teaching

    func teach(sessionId: String, resolutions: [(editId: String, accepted: Bool)]) async {
        let editResolutions = resolutions.map { r in
            EditResolution(editId: r.editId, accepted: r.accepted)
        }
        do {
            try await engine.correctTeach(sessionId: sessionId, resolutions: editResolutions)
        } catch {
            logger.error("teach error: \(error)")
        }
    }

    func save() async {
        do {
            try await engine.correctSave()
        } catch {
            logger.error("save error: \(error)")
        }
    }
}
