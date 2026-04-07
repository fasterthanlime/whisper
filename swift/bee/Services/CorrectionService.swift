import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "CorrectionService")

final class CorrectionService: @unchecked Sendable {
    private var engine: OpaquePointer?  // CorrectionEngine*
    private let lock = NSLock()

    struct Edit: Codable {
        let edit_id: String
        let span_start: Int
        let span_end: Int
        let original: String
        let replacement: String
        let term: String
        let alias_id: Int?
        let ranker_prob: Double
        let gate_prob: Double
    }

    struct EditsEnvelope: Codable {
        let session_id: String
        let edits: [Edit]
    }

    struct Output: Sendable {
        let sessionId: String
        let originalText: String
        let bestText: String
        let edits: [Edit]
    }

    enum LoadError: LocalizedError {
        case failed(String)
        var errorDescription: String? {
            switch self {
            case .failed(let msg): "Correction engine load failed: \(msg)"
            }
        }
    }

    // MARK: - Lifecycle

    func load(
        datasetDir: String, eventsPath: String?,
        gateThreshold: Float = 0.5, rankerThreshold: Float = 0.2
    ) async throws {
        unload()

        let loaded: OpaquePointer = try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                var err: UnsafeMutablePointer<CChar>?
                let ptr = datasetDir.withCString { dsPtr in
                    if let eventsPath {
                        return eventsPath.withCString { evPtr in
                            bee_correct_engine_load(dsPtr, evPtr, gateThreshold, rankerThreshold, &err)
                        }
                    } else {
                        return bee_correct_engine_load(dsPtr, nil, gateThreshold, rankerThreshold, &err)
                    }
                }
                if let ptr {
                    cont.resume(returning: ptr)
                } else {
                    let msg = err.flatMap { String(cString: $0) } ?? "unknown"
                    if let err { asr_string_free(err) }
                    cont.resume(throwing: LoadError.failed(msg))
                }
            }
        }

        lock.withLock { engine = loaded }
        logger.info("Correction engine loaded")
    }

    func unload() {
        let eng = lock.withLock { () -> OpaquePointer? in
            let e = engine; engine = nil; return e
        }
        if let eng { bee_correct_engine_free(eng) }
    }

    deinit { if let engine { bee_correct_engine_free(engine) } }

    // MARK: - Processing

    func process(text: String, appId: String?) -> Output? {
        guard let eng = lock.withLock({ engine }) else { return nil }

        // Build input JSON
        var dict: [String: Any] = ["text": text]
        if let appId { dict["app_id"] = appId }
        guard let jsonData = try? JSONSerialization.data(withJSONObject: dict),
              let jsonStr = String(data: jsonData, encoding: .utf8)
        else { return nil }

        var err: UnsafeMutablePointer<CChar>?
        let result = jsonStr.withCString { bee_correct_process(eng, $0, &err) }
        guard let result else {
            if let err {
                logger.error("process error: \(String(cString: err), privacy: .public)")
                asr_string_free(err)
            }
            return nil
        }
        defer { bee_correction_result_free(result) }

        let sessionIdPtr = bee_correction_result_session_id(result)
        let bestTextPtr = bee_correction_result_best_text(result)
        let editsJsonPtr = bee_correction_result_json(result)

        let sessionId = sessionIdPtr.flatMap { String(cString: $0) } ?? ""
        let bestText = bestTextPtr.flatMap { String(cString: $0) } ?? text
        let editsJsonStr = editsJsonPtr.flatMap { String(cString: $0) } ?? "{}"

        if let sessionIdPtr { asr_string_free(sessionIdPtr) }
        if let bestTextPtr { asr_string_free(bestTextPtr) }
        // editsJsonPtr is a const char* — NOT freed (owned by result)

        // Parse edits
        var edits: [Edit] = []
        if let data = editsJsonStr.data(using: .utf8),
           let envelope = try? JSONDecoder().decode(EditsEnvelope.self, from: data)
        {
            edits = envelope.edits
        }

        if edits.isEmpty {
            return nil  // no corrections to make
        }

        logger.info("Corrected \(edits.count) span(s): \"\(text.prefix(40))\" → \"\(bestText.prefix(40))\"")

        return Output(
            sessionId: sessionId,
            originalText: text,
            bestText: bestText,
            edits: edits
        )
    }

    // MARK: - Teaching

    func teach(sessionId: String, resolutions: [(editId: String, accepted: Bool)]) {
        guard let eng = lock.withLock({ engine }) else { return }

        let editsArray = resolutions.map { r in
            ["edit_id": r.editId, "resolution": r.accepted ? "accepted" : "rejected"]
        }
        let dict: [String: Any] = ["edits": editsArray]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: dict),
              let jsonStr = String(data: jsonData, encoding: .utf8)
        else { return }

        var err: UnsafeMutablePointer<CChar>?
        sessionId.withCString { sid in
            jsonStr.withCString { json in
                bee_correct_teach(eng, sid, json, &err)
            }
        }
        if let err {
            logger.error("teach error: \(String(cString: err), privacy: .public)")
            asr_string_free(err)
        }
    }

    func save() {
        guard let eng = lock.withLock({ engine }) else { return }
        var err: UnsafeMutablePointer<CChar>?
        bee_correct_save(eng, &err)
        if let err {
            logger.error("save error: \(String(cString: err), privacy: .public)")
            asr_string_free(err)
        }
    }
}

// Sendable conformance for Edit (all properties are value types)
extension CorrectionService.Edit: Sendable {}
