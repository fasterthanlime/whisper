import Foundation

/// Downloads HuggingFace model files with progress reporting.
/// Progress is reported per-file (0→1 for each file). Small config files
/// flash through instantly; the large safetensors files show real progress
/// based on the HTTP Content-Length header.
final class HFDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var continuation: CheckedContinuation<Void, Error>?
    private var onProgress: ((Double) -> Void)?
    private var destinationURL: URL?
    private lazy var session: URLSession = {
        URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }()

    /// Download all missing repos to the cache directory.
    /// Returns the number of files downloaded.
    static func downloadMissing(
        repos: [RepoDownload],
        cacheDir: URL,
        onProgress: @escaping (_ progress: Double, _ model: String) -> Void
    ) async throws -> Int {
        var totalDownloaded = 0
        for repo in repos {
            let repoDir = cacheDir.appendingPathComponent(repo.localDir)
            try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
            let modelName = repo.repoId.components(separatedBy: "/").last ?? repo.repoId

            for file in repo.files {
                let dest = repoDir.appendingPathComponent(file.name)
                if FileManager.default.fileExists(atPath: dest.path) { continue }

                let downloader = HFDownloader()
                try await downloader.download(
                    from: file.url,
                    to: dest
                ) { fileProgress in
                    onProgress(fileProgress, modelName)
                }
                totalDownloaded += 1
            }
        }

        return totalDownloaded
    }

    private func download(
        from urlString: String,
        to destination: URL,
        onProgress: @escaping (Double) -> Void
    ) async throws {
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }

        self.onProgress = onProgress
        self.destinationURL = destination

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.continuation = cont
            let task = self.session.downloadTask(with: url)
            task.resume()
        }
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let destination = destinationURL else {
            continuation?.resume(throwing: URLError(.cannotCreateFile))
            continuation = nil
            return
        }

        // Check HTTP status code
        if let httpResponse = downloadTask.response as? HTTPURLResponse,
           !(200..<300).contains(httpResponse.statusCode) {
            continuation?.resume(throwing: URLError(.init(rawValue: httpResponse.statusCode),
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode) for \(downloadTask.originalRequest?.url?.absoluteString ?? "unknown")"]))
            continuation = nil
            return
        }

        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            continuation?.resume()
        } catch {
            continuation?.resume(throwing: error)
        }
        continuation = nil
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        if totalBytesExpectedToWrite > 0 {
            let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            onProgress?(progress)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
