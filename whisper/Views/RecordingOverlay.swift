import SwiftUI

/// Result state for overlay dismissal animation
enum OverlayResult {
    case none
    case success
    case cancelled
}

/// Floating overlay showing transcript with spectrum bar below.
struct RecordingOverlayView: View {
    let appState: AppState
    let result: OverlayResult

    @State private var isAppearing = false
    @State private var displayedText = ""
    @State private var textAnimationTask: Task<Void, Never>?

    var body: some View {
        ZStack {
            // Main content
            if result == .none {
                mainContent
                    .scaleEffect(isAppearing ? 1.0 : 0.8)
                    .opacity(isAppearing ? 1.0 : 0.0)
            }

            // Success overlay
            if result == .success {
                resultOverlay(success: true)
            }

            // Cancelled overlay
            if result == .cancelled {
                resultOverlay(success: false)
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isAppearing)
        .animation(.easeInOut(duration: 0.2), value: result)
        .onAppear {
            withAnimation {
                isAppearing = true
            }
        }
        .onChange(of: appState.partialTranscript) { _, newValue in
            animateTextChange(to: newValue)
        }
    }

    private var mainContent: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Transcript text with typewriter effect
            Text(displayedTextValue)
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(.white)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .topLeading)

            // Spectrum bar below text
            SpectrumBarView(bands: appState.spectrumBands, isActive: appState.phase == .recording)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 14)
        .frame(width: 500, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.black.opacity(0.8))
                .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
        )
    }

    private func resultOverlay(success: Bool) -> some View {
        HStack(spacing: 12) {
            Image(systemName: success ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.system(size: 28, weight: .medium))
                .foregroundColor(success ? .green : .red)
                .symbolEffect(.bounce, value: result)

            Text(success ? "Pasted" : "Cancelled")
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(.white)
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 20)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.black.opacity(0.8))
                .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
        )
        .transition(.scale.combined(with: .opacity))
    }

    private var displayedTextValue: String {
        if displayedText.isEmpty && appState.partialTranscript.isEmpty {
            return appState.phase == .recording ? "Listening..." : "Transcribing..."
        }
        return displayedText.isEmpty ? appState.partialTranscript : displayedText
    }

    private func animateTextChange(to newText: String) {
        textAnimationTask?.cancel()

        // Find the common prefix - only animate new characters
        let currentText = displayedText
        let commonPrefixLength = currentText.commonPrefix(with: newText).count

        if commonPrefixLength == newText.count {
            // New text is shorter or same - just set it
            displayedText = newText
            return
        }

        // Animate new characters appearing
        let newPart = String(newText.dropFirst(commonPrefixLength))
        displayedText = String(newText.prefix(commonPrefixLength))

        textAnimationTask = Task { @MainActor in
            for char in newPart {
                guard !Task.isCancelled else { return }
                displayedText.append(char)
                try? await Task.sleep(for: .milliseconds(15))
            }
        }
    }
}

/// Thin spectrum bar showing audio activity.
struct SpectrumBarView: View {
    let bands: [Float]
    let isActive: Bool

    private let barCount = 32
    private let barHeight: CGFloat = 4

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 1) {
                ForEach(0..<barCount, id: \.self) { index in
                    let bandIndex = index * bands.count / barCount
                    let level = bandIndex < bands.count ? CGFloat(bands[bandIndex]) : 0

                    RoundedRectangle(cornerRadius: 1)
                        .fill(Color.white.opacity(isActive ? 0.3 + 0.7 * level : 0.2))
                        .frame(height: barHeight)
                }
            }
            .frame(width: geo.size.width, height: barHeight)
        }
        .frame(height: barHeight)
        .animation(.easeOut(duration: 0.05), value: bands)
    }
}
