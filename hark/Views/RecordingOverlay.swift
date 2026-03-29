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
    @State private var freshStart = 0 // character index where fresh/rewritten text begins
    @State private var freshOpacity: Double = 1.0
    @State private var freshColorBlend: Double = 1.0
    @State private var textAnimationTask: Task<Void, Never>?
    @State private var textContentHeight: CGFloat = 0

    private var dismissResult: OverlayResult { appState.overlayDismiss }

    private var scale: CGFloat {
        1.0
    }

    private var opacity: Double {
        if dismissResult != .none { return 0 }
        return isAppearing ? 1.0 : 0.0
    }

    private var introOffsetY: CGFloat {
        0
    }

    private var introBlurRadius: CGFloat {
        0
    }

    private let footerBandHeight: CGFloat = 36
    private let footerOutsideGap: CGFloat = 6
    private let pendingOpacity: Double = 0.74
    private let freshStartOpacity: Double = 0.5

    var body: some View {
        GeometryReader { geo in
            controlFrameLayout(size: geo.size)
            .scaleEffect(scale)
            .offset(y: introOffsetY)
            .blur(radius: introBlurRadius)
            .opacity(opacity)
            .animation(.easeOut(duration: 0.18), value: isAppearing)
            .animation(.easeIn(duration: 0.25), value: dismissResult)
        }
        .onAppear {
            withAnimation {
                isAppearing = true
            }
        }
        .onChange(of: appState.partialTranscript) { _, newValue in
            animateTextChange(to: newValue)
        }
    }

    @ViewBuilder
    private func controlFrameLayout(size: CGSize) -> some View {
        let reservedHeight = footerBandHeight + footerOutsideGap
        let textViewportHeight = max(24, size.height - reservedHeight)
        VStack(spacing: 0) {
            if appState.overlayFooterAbove {
                footerBand
                    .padding(.horizontal, 6)
                    .padding(.bottom, footerOutsideGap)
            }

            transcriptBody(maxTextHeight: textViewportHeight)
                .frame(maxWidth: .infinity, maxHeight: textViewportHeight, alignment: .topLeading)

            if !appState.overlayFooterAbove {
                footerBand
                    .padding(.horizontal, 6)
                    .padding(.top, footerOutsideGap)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func transcriptBody(maxTextHeight: CGFloat) -> some View {
        return ScrollViewReader { proxy in
            ScrollView(.vertical, showsIndicators: false) {
                StyledTranscriptText(
                    fullText: displayedTextValue,
                    committedUTF16: appState.partialTranscriptCommittedUTF16,
                    freshStart: freshStart,
                    freshOpacity: freshOpacity,
                    freshColorBlend: freshColorBlend
                )
                .equatable()
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .background(GeometryReader { geo in
                        Color.clear.preference(key: TextHeightKey.self, value: geo.size.height)
                    })
                    .id("transcript")
            }
            .frame(maxHeight: maxTextHeight, alignment: .top)
            .mask(Color.white)
            .onPreferenceChange(TextHeightKey.self) { textContentHeight = $0 }
            .onChange(of: displayedText) { _, _ in
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("transcript", anchor: .bottom)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.black.opacity(0.94))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                    )
            )
        }
    }

    private var footerBand: some View {
        HStack(alignment: .center) {
            waveformBlock

            Spacer(minLength: 8)

            controlsBlock
                .animation(.easeInOut(duration: 0.15), value: appState.isLockedMode)
                .animation(.easeInOut(duration: 0.15), value: appState.isFinishing)
        }
        .padding(.horizontal, 2)
        .frame(height: footerBandHeight)
    }

    private var waveformBlock: some View {
        Group {
            if appState.isFinishing {
                ThinkingBarsView()
            } else {
                SpectrumBarsView(bands: appState.spectrumBands)
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 28)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.black.opacity(0.82))
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
                )
        )
    }

    private var controlsBlock: some View {
        hintView
            .lineLimit(1)
            .padding(.horizontal, 10)
            .frame(height: 28)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.black.opacity(0.82))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
                    )
            )
    }

    // MARK: - Hints

    @ViewBuilder
    private var hintView: some View {
        if appState.overlayTetherOutOfApp {
            let appName = appState.overlayLockedAppName ?? "original app"
            HStack(spacing: 6) {
                keyCap("🔒")
                hintLabel("Recording locked to \(appName)")
            }
        } else
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

    private func animateTextChange(to newText: String) {
        guard !newText.isEmpty else { return }

        textAnimationTask?.cancel()

        // After final inference, show complete text instantly.
        if appState.phase == .transcribing {
            displayedText = newText
            freshStart = 0
            freshOpacity = pendingOpacity
            freshColorBlend = 1.0
            return
        }

        let currentText = displayedText
        let commonPrefixLength = currentText.commonPrefix(with: newText).count

        // Text unchanged or only shortened — snap immediately.
        if commonPrefixLength == newText.count {
            displayedText = newText
            freshStart = 0
            freshOpacity = pendingOpacity
            freshColorBlend = 1.0
            return
        }

        let newPart = String(newText.dropFirst(commonPrefixLength))

        // Set the full new text, mark where fresh content starts.
        freshStart = commonPrefixLength
        freshOpacity = freshStartOpacity
        freshColorBlend = 0.0
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
                freshOpacity = pendingOpacity
                freshColorBlend = 1.0
            }
        }
    }

}

private struct StyledTranscriptText: View, Equatable {
    let fullText: String
    let committedUTF16: Int
    let freshStart: Int
    let freshOpacity: Double
    let freshColorBlend: Double

    private static let transcriptFont: NSFont =
        NSFont(name: "Jost-Regular", size: 15.2) ?? .systemFont(ofSize: 15.2, weight: .regular)
    private static let committedRGB = (r: 0.93, g: 0.93, b: 0.92)
    private static let pendingRGB = (r: 0.80, g: 0.79, b: 0.76)
    private static let freshRGB = (r: 0.95, g: 0.72, b: 0.54)
    private static let pendingOpacity: Double = 0.74

    static func == (lhs: StyledTranscriptText, rhs: StyledTranscriptText) -> Bool {
        lhs.fullText == rhs.fullText
            && lhs.committedUTF16 == rhs.committedUTF16
            && lhs.freshStart == rhs.freshStart
            && lhs.freshOpacity == rhs.freshOpacity
            && lhs.freshColorBlend == rhs.freshColorBlend
    }

    var body: some View {
        Text(styledText)
    }

    private var styledText: AttributedString {
        var result = AttributedString(fullText)
        let committedCount = committedCharacterCount()
        let pendingStart = min(max(committedCount, 0), fullText.count)
        let freshStartClamped = min(max(freshStart, pendingStart), fullText.count)
        let committedColor = Color(red: Self.committedRGB.r, green: Self.committedRGB.g, blue: Self.committedRGB.b)
        let pendingColor = Color(red: Self.pendingRGB.r, green: Self.pendingRGB.g, blue: Self.pendingRGB.b)
        let freshColor = blendedFreshColor()

        result.font = Self.transcriptFont
        result.foregroundColor = pendingColor.opacity(Self.pendingOpacity)

        if committedCount > 0 {
            let committedEnd = result.index(result.startIndex, offsetByCharacters: committedCount)
            result[result.startIndex..<committedEnd].foregroundColor = committedColor
        }

        if freshStartClamped < fullText.count {
            let startIdx = result.index(result.startIndex, offsetByCharacters: freshStartClamped)
            result[startIdx..<result.endIndex].foregroundColor = freshColor.opacity(freshOpacity)
        }

        return result
    }

    private func committedCharacterCount() -> Int {
        guard !fullText.isEmpty else { return 0 }
        let clampedUTF16 = min(max(0, committedUTF16), (fullText as NSString).length)
        let ns = fullText as NSString
        return ns.substring(to: clampedUTF16).count
    }

    private func blendedFreshColor() -> Color {
        let t = min(max(freshColorBlend, 0), 1)
        let r = Self.freshRGB.r + (Self.pendingRGB.r - Self.freshRGB.r) * t
        let g = Self.freshRGB.g + (Self.pendingRGB.g - Self.freshRGB.g) * t
        let b = Self.freshRGB.b + (Self.pendingRGB.b - Self.freshRGB.b) * t
        return Color(red: r, green: g, blue: b)
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
