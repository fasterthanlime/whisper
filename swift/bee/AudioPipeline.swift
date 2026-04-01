import Foundation

/// A multi-producer, single-consumer channel.
///
/// The producer calls `send(_:)` (non-blocking).
/// The consumer iterates `stream` to be woken, then calls `drain()`
/// to get all queued items at once — natural batching when the consumer
/// is slower than the producer.
final class Pipeline<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: [T] = []
    private var continuation: AsyncStream<Void>.Continuation?
    private var finished = false

    let stream: AsyncStream<Void>

    init() {
        var cont: AsyncStream<Void>.Continuation?
        stream = AsyncStream<Void>(bufferingPolicy: .bufferingNewest(1)) { c in
            cont = c
        }
        continuation = cont
    }

    /// Non-blocking send. Called by producers.
    func send(_ item: T) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        buffer.append(item)
        let cont = continuation
        lock.unlock()
        cont?.yield(())
    }

    /// Close the stream. No more items accepted.
    /// The consumer will see remaining items on next drain, then the
    /// stream ends.
    func finish() {
        lock.lock()
        finished = true
        let cont = continuation
        continuation = nil
        lock.unlock()
        cont?.finish()
    }

    /// True if the buffer has pending items. Non-blocking peek for the consumer.
    var isEmpty: Bool {
        lock.lock()
        let empty = buffer.isEmpty
        lock.unlock()
        return empty
    }

    /// Returns all queued items at once. Called by the consumer after
    /// being woken by the stream.
    func drain() -> [T] {
        lock.lock()
        let items = buffer
        buffer.removeAll(keepingCapacity: true)
        if finished {
            let cont = continuation
            continuation = nil
            lock.unlock()
            cont?.finish()
        } else {
            lock.unlock()
        }
        return items
    }
}

// MARK: - Channel types

/// Channel 0: AudioEngine → Capture Task.
/// Just resampled 16kHz audio. No end signal. Continuous while warm.
typealias RawAudioPipeline = Pipeline<[Float]>

/// Channel 1: Capture Task → ASR Task.
enum AudioChunk: Sendable {
    case samples([Float])
    case end(EndMode)
}
typealias AudioPipeline = Pipeline<AudioChunk>

/// Channel 2: ASR Task → Session actor.
enum TranscriptEvent: Sendable {
    case partial(String)
    case done(text: String, mode: EndMode)
}
typealias TranscriptPipeline = Pipeline<TranscriptEvent>

/// How a session ends. Flows forward through all three channels.
enum EndMode: Sendable {
    case commit(submit: Bool)
    case cancel
}
