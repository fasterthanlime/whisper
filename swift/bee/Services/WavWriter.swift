import Foundation

enum WavWriter {
    /// Write 16kHz mono float32 samples as a 16-bit PCM WAV file.
    static func write(samples: [Float], sampleRate: Int = 16_000, to url: URL) throws {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count * Int(blockAlign))

        var data = Data()

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(uint32LE: 36 + dataSize)
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(uint32LE: 16) // chunk size
        data.append(uint16LE: 1) // PCM format
        data.append(uint16LE: numChannels)
        data.append(uint32LE: UInt32(sampleRate))
        data.append(uint32LE: byteRate)
        data.append(uint16LE: blockAlign)
        data.append(uint16LE: bitsPerSample)

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(uint32LE: dataSize)

        // Convert float32 → int16
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            data.append(uint16LE: UInt16(bitPattern: int16))
        }

        try data.write(to: url)
    }
}

private extension Data {
    mutating func append(uint16LE value: UInt16) {
        var v = value.littleEndian
        append(UnsafeBufferPointer(start: &v, count: 1))
    }

    mutating func append(uint32LE value: UInt32) {
        var v = value.littleEndian
        append(UnsafeBufferPointer(start: &v, count: 1))
    }
}
