import SwiftUI

/// Floating overlay showing transcript with inline spectrum visualizer.
struct RecordingOverlayView: View {
    let appState: AppState

    var body: some View {
        if isVisible {
            HStack(alignment: .bottom, spacing: 0) {
                // Transcript text (if any)
                if !appState.partialTranscript.isEmpty {
                    Text(appState.partialTranscript)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(.white)
                        .lineLimit(3)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Inline spectrum visualizer
                InlineSpectrumView(bands: appState.spectrumBands, isActive: appState.phase == .recording)
                    .padding(.leading, appState.partialTranscript.isEmpty ? 0 : 6)
            }
            .padding(.vertical, 10)
            .padding(.horizontal, 14)
            .frame(maxWidth: 550, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .shadow(color: .black.opacity(0.2), radius: 8, y: 4)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(.white.opacity(0.1), lineWidth: 1)
            )
        }
    }

    private var isVisible: Bool {
        appState.phase == .recording || appState.phase == .transcribing
    }
}

/// Compact inline spectrum visualizer that looks like a cursor.
struct InlineSpectrumView: View {
    let bands: [Float]
    let isActive: Bool

    private let barCount = 8
    private let barWidth: CGFloat = 3
    private let barSpacing: CGFloat = 2
    private let maxHeight: CGFloat = 18
    private let minHeight: CGFloat = 3

    var body: some View {
        HStack(alignment: .bottom, spacing: barSpacing) {
            ForEach(0..<barCount, id: \.self) { index in
                let level = index < bands.count ? CGFloat(bands[index]) : 0
                let height = minHeight + (maxHeight - minHeight) * level

                RoundedRectangle(cornerRadius: 1.5)
                    .fill(barGradient)
                    .frame(width: barWidth, height: isActive ? height : minHeight)
                    .animation(.easeOut(duration: 0.08), value: level)
            }
        }
        .opacity(isActive ? 1.0 : 0.4)
        .animation(.easeInOut(duration: 0.15), value: isActive)
    }

    private var barGradient: LinearGradient {
        LinearGradient(
            colors: [
                Color(hue: 0.55, saturation: 0.8, brightness: 1.0),  // cyan
                Color(hue: 0.75, saturation: 0.7, brightness: 1.0)   // purple
            ],
            startPoint: .bottom,
            endPoint: .top
        )
    }
}
