import Foundation

/// Downloads HuggingFace model files with progress reporting.
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
        onProgress: @escaping (Double) -> Void
    ) async throws -> Int {
        // Collect all files that need downloading
        var filesToDownload: [(url: String, destination: URL, size: UInt64)] = []
        for repo in repos {
            let repoDir = cacheDir.appendingPathComponent(repo.localDir)
            try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
            for file in repo.files {
                let dest = repoDir.appendingPathComponent(file.name)
                if !FileManager.default.fileExists(atPath: dest.path) {
                    filesToDownload.append((url: file.url, destination: dest, size: file.size))
                }
            }
        }

        if filesToDownload.isEmpty { return 0 }

        let totalBytes = filesToDownload.reduce(UInt64(0)) { $0 + max($1.size, 1) }
        var completedBytes: UInt64 = 0

        for file in filesToDownload {
            let fileSize = Double(max(file.size, 1))
            let baseBytes = Double(completedBytes)

            let downloader = HFDownloader()
            try await downloader.download(
                from: file.url,
                to: file.destination
            ) { fileProgress in
                onProgress((baseBytes + fileProgress * fileSize) / Double(totalBytes))
            }

            completedBytes += max(file.size, 1)
        }

        onProgress(1.0)
        return filesToDownload.count
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
            // Remove existing file if any
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
