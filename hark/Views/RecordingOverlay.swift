import SwiftUI

/// Result state for overlay dismissal animation.
enum OverlayResult {
    case none
    case success
    case cancelled
}

/// Floating overlay showing transcript with spectrum bars and contextual hints.
struct RecordingOverlayView: View {
    let appState: AppState

    @State private var isAppearing = false
    @State private var displayedText = ""
    @State private var stablePrefix = ""
    @State private var freshStart = 0 // character index where fresh/rewritten text begins
    @State private var freshOpacity: Double = 1.0
    @State private var textAnimationTask: Task<Void, Never>?
    @State private var textContentHeight: CGFloat = 0

    private var dismissResult: OverlayResult { appState.overlayDismiss }

    private var scale: CGFloat {
        if dismissResult == .success { return 1.3 }
        if dismissResult == .cancelled { return 0.7 }
        return isAppearing ? 1.0 : 0.8
    }

    private var opacity: Double {
        if dismissResult != .none { return 0 }
        return isAppearing ? 1.0 : 0.0
    }

    private let maxTextHeight: CGFloat = 120 // ~6 lines at 17pt

    private var isScrolling: Bool { textContentHeight > maxTextHeight }

    var body: some View {
        mainContent
            .scaleEffect(scale)
            .opacity(opacity)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isAppearing)
            .animation(.easeIn(duration: 0.25), value: dismissResult)
            .frame(width: 700, height: 300, alignment: .top)
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
        VStack(spacing: 0) {
            // Text area with scroll
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: false) {
                    Text(styledDisplayText)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .background(GeometryReader { geo in
                            Color.clear.preference(key: TextHeightKey.self, value: geo.size.height)
                        })
                        .id("transcript")
                }
                .frame(maxHeight: maxTextHeight, alignment: .top)
                .mask(isScrolling ? textScrollMask : AnyView(Color.white))
                .onPreferenceChange(TextHeightKey.self) { textContentHeight = $0 }
                .onChange(of: displayedText) { _, _ in
                    withAnimation(.easeOut(duration: 0.1)) {
                        proxy.scrollTo("transcript", anchor: .bottom)
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)
            .padding(.bottom, 10)

            // Bottom bar: indicator on left, hints on right — closer to edges
            HStack(alignment: .center) {
                if appState.isFinishing {
                    ThinkingBarsView()
                } else {
                    SpectrumBarsView(bands: appState.spectrumBands)
                }

                Spacer()

                hintView
                    .animation(.easeInOut(duration: 0.15), value: appState.isLockedMode)
                    .animation(.easeInOut(duration: 0.15), value: appState.isFinishing)
            }
            .padding(.leading, 20)
            .padding(.trailing, 14)
            .padding(.bottom, 12)
        }
        .frame(width: 500, alignment: .topLeading)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.ultraThinMaterial)
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.black.opacity(0.8))
            }
            .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
        )
    }

    // Gradient mask that fades the top when content is scrolling
    private var textScrollMask: AnyView {
        AnyView(
            VStack(spacing: 0) {
                LinearGradient(
                    colors: [.clear, .white],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 20)
                Color.white
            }
        )
    }

    // MARK: - Hints

    @ViewBuilder
    private var hintView: some View {
        if appState.isFinishing {
            hintLabel("Finalizing...")
        } else if appState.isLockedMode {
            let key = appState.hotkeyBinding.displayLabel
            HStack(spacing: 4) {
                keyCap(key)
                hintLabel("to submit ·")
                keyCap(key)
                hintLabel("+")
                keyCap("Esc")
                hintLabel("to cancel")
            }
        } else {
            HStack(spacing: 4) {
                hintLabel("Release to submit ·")
                keyCap("⌘")
                hintLabel("to lock ·")
                keyCap("Esc")
                hintLabel("to cancel")
            }
        }
    }

    private func keyCap(_ label: String) -> some View {
        Text(label)
            .font(.system(.caption2, weight: .medium).monospaced())
            .foregroundColor(.white.opacity(0.7))
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(Color.white.opacity(0.12))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.2), lineWidth: 0.5)
            )
    }

    private func hintLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(.caption2, weight: .medium))
            .foregroundColor(.white.opacity(0.45))
    }

    // MARK: - Text

    private var displayedTextValue: String {
        if !displayedText.isEmpty { return displayedText }
        if !appState.partialTranscript.isEmpty { return appState.partialTranscript }
        return "Listening..."
    }

    /// Build an `AttributedString` where the stable prefix is fully opaque
    /// and the fresh/rewritten portion fades in.
    private var styledDisplayText: AttributedString {
        let full = displayedTextValue
        var result = AttributedString(full)
        let font = NSFont(name: "Jost-Medium", size: 15) ?? .systemFont(ofSize: 15, weight: .medium)

        result.font = font
        result.foregroundColor = .white

        // Apply fade to the fresh portion (after the stable prefix)
        if freshStart > 0, freshStart < full.count {
            let startIdx = result.index(result.startIndex, offsetByCharacters: freshStart)
            result[startIdx..<result.endIndex].foregroundColor = .white.opacity(freshOpacity)
        }

        return result
    }

    private func animateTextChange(to newText: String) {
        guard !newText.isEmpty else { return }

        textAnimationTask?.cancel()

        // After final inference, show complete text instantly.
        if appState.phase == .transcribing {
            displayedText = newText
            freshStart = 0
            freshOpacity = 1.0
            return
        }

        let currentText = displayedText
        let commonPrefixLength = currentText.commonPrefix(with: newText).count

        // Text unchanged or only shortened — snap immediately.
        if commonPrefixLength == newText.count {
            displayedText = newText
            freshStart = 0
            freshOpacity = 1.0
            return
        }

        let hasRewrite = commonPrefixLength < currentText.count
        let newPart = String(newText.dropFirst(commonPrefixLength))

        // Set the full new text, mark where fresh content starts.
        freshStart = commonPrefixLength
        freshOpacity = hasRewrite ? 0.35 : 0.35
        displayedText = String(newText.prefix(commonPrefixLength))

        // Typewrite the new characters, then fade in the fresh portion.
        textAnimationTask = Task { @MainActor in
            // First: if there's rewritten text, show it all at once at low opacity
            // (it's already set via freshStart/freshOpacity above)

            // Typewrite the genuinely new characters
            let charCount = newPart.count
            let delayMs = max(5, min(50, 800 / max(charCount, 1)))
            for char in newPart {
                guard !Task.isCancelled else { return }
                displayedText.append(char)
                try? await Task.sleep(for: .milliseconds(delayMs))
            }

            // Fade in the fresh portion
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.2)) {
                freshOpacity = 1.0
            }
        }
    }
}

// MARK: - Preference Key for text height measurement

private struct TextHeightKey: PreferenceKey {
    nonisolated(unsafe) static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

// MARK: - Spectrum Bars (no background circle)

/// Six vertical capsule bars, tapered at the edges.
struct SpectrumBarsView: View {
    let bands: [Float]

    private let barCount = 6
    private let barWidth: CGFloat = 2.5
    private let spacing: CGFloat = 2
    private let maxBarHeight: CGFloat = 18
    private let taperFactors: [CGFloat] = [0.45, 0.75, 1.0, 1.0, 0.75, 0.45]

    var body: some View {
        HStack(alignment: .center, spacing: spacing) {
            ForEach(0..<barCount, id: \.self) { index in
                let level = index < bands.count ? CGFloat(bands[index]) : 0
                let taper = taperFactors[index]
                let maxH = maxBarHeight * taper
                let minH: CGFloat = 3
                let height = minH + (maxH - minH) * level

                Capsule()
                    .fill(Color.white.opacity(0.4 + 0.5 * level))
                    .frame(width: barWidth, height: height)
            }
        }
        .animation(.easeOut(duration: 0.07), value: bands)
        .frame(height: maxBarHeight)
    }
}

// MARK: - Thinking Bars (animated wave for finalizing)

/// Animated wave bars that pulse to indicate processing.
struct ThinkingBarsView: View {
    @State private var phase: CGFloat = 0

    private let barCount = 6
    private let barWidth: CGFloat = 2.5
    private let spacing: CGFloat = 2
    private let maxBarHeight: CGFloat = 18

    var body: some View {
        HStack(alignment: .center, spacing: spacing) {
            ForEach(0..<barCount, id: \.self) { index in
                let offset = Double(index) / Double(barCount) * .pi * 2
                let level = (sin(phase + offset) + 1) / 2

                Capsule()
                    .fill(Color.white.opacity(0.3 + 0.4 * level))
                    .frame(width: barWidth, height: 3 + (maxBarHeight - 3) * level)
            }
        }
        .frame(height: maxBarHeight)
        .onAppear {
            withAnimation(.linear(duration: 0.8).repeatForever(autoreverses: false)) {
                phase = .pi * 2
            }
        }
    }
}
