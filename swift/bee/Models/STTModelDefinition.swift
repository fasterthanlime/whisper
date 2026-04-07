import Foundation

enum ModelFormat: Hashable {
    case safetensors
    case gguf(repoID: String, filename: String, baseRepoID: String)
}

struct STTModelDefinition: Identifiable, Hashable {
    let id: String
    let displayName: String
    let repoID: String
    let format: ModelFormat
    let downloadSizeMB: Int

    static let allModels: [STTModelDefinition] = [
        STTModelDefinition(
            id: "qwen3-0.6b",
            displayName: "Qwen3 ASR 0.6B",
            repoID: "Qwen/Qwen3-ASR-0.6B",
            format: .safetensors,
            downloadSizeMB: 1200
        ),
        STTModelDefinition(
            id: "qwen3-0.6b-q8",
            displayName: "Qwen3 ASR 0.6B (Q8_0)",
            repoID: "Qwen/Qwen3-ASR-0.6B",
            format: .gguf(
                repoID: "Alkd/qwen3-asr-gguf",
                filename: "qwen3_asr_0.6b_q8_0.gguf",
                baseRepoID: "Qwen/Qwen3-ASR-0.6B"
            ),
            downloadSizeMB: 1010
        ),
        STTModelDefinition(
            id: "qwen3-0.6b-q4k",
            displayName: "Qwen3 ASR 0.6B (Q4_K)",
            repoID: "Qwen/Qwen3-ASR-0.6B",
            format: .gguf(
                repoID: "Alkd/qwen3-asr-gguf",
                filename: "qwen3_asr_0.6b_q4_k.gguf",
                baseRepoID: "Qwen/Qwen3-ASR-0.6B"
            ),
            downloadSizeMB: 605
        ),
        STTModelDefinition(
            id: "qwen3-1.7b",
            displayName: "Qwen3 ASR 1.7B",
            repoID: "Qwen/Qwen3-ASR-1.7B",
            format: .safetensors,
            downloadSizeMB: 3400
        ),
        STTModelDefinition(
            id: "qwen3-1.7b-q8",
            displayName: "Qwen3 ASR 1.7B (Q8_0)",
            repoID: "Qwen/Qwen3-ASR-1.7B",
            format: .gguf(
                repoID: "Alkd/qwen3-asr-gguf",
                filename: "qwen3_asr_1.7b_q8_0.gguf",
                baseRepoID: "Qwen/Qwen3-ASR-1.7B"
            ),
            downloadSizeMB: 2510
        ),
        STTModelDefinition(
            id: "qwen3-1.7b-q4k",
            displayName: "Qwen3 ASR 1.7B (Q4_K)",
            repoID: "Qwen/Qwen3-ASR-1.7B",
            format: .gguf(
                repoID: "Alkd/qwen3-asr-gguf",
                filename: "qwen3_asr_1.7b_q4_k.gguf",
                baseRepoID: "Qwen/Qwen3-ASR-1.7B"
            ),
            downloadSizeMB: 1340
        ),
        STTModelDefinition(
            id: "qwen3-1.7b-mlx-4bit",
            displayName: "Qwen3 ASR 1.7B (MLX 4-bit)",
            repoID: "mlx-community/Qwen3-ASR-1.7B-4bit",
            format: .safetensors,
            downloadSizeMB: 1100
        ),
    ]

    static let `default` = allModels[0]

    var cacheDirName: String {
        switch format {
        case .safetensors:
            return repoID.replacingOccurrences(of: "/", with: "--")
        case .gguf(let ggufRepoID, let ggufFilename, _):
            let sanitizedRepo = ggufRepoID.replacingOccurrences(of: "/", with: "--")
            let sanitizedFile = ggufFilename.replacingOccurrences(of: ".", with: "_")
            return "\(sanitizedRepo)--\(sanitizedFile)"
        }
    }

    static var cacheDirectory: String {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return caches.appendingPathComponent("models").path
    }
}
