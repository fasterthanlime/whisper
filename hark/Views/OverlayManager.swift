import AppKit
import os
import SwiftUI

/// Manages the lifecycle of floating recording indicator panels — one per screen.
@MainActor
final class OverlayManager {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "OverlayManager"
    )

    private enum AnchorSource: String {
        case caret = "caret"
        case focusedElementFrame = "focused_element_frame"
        case focusedWindowFrame = "focused_window_frame"
        case mouse = "mouse"
    }

    private enum AnchorFallbackReason: String {
        case axNotTrusted = "ax_not_trusted"
        case noFocusedElement = "no_focused_element"
        case noAccessibleGeometry = "no_accessible_geometry"
    }

    private enum OverlayPlacement: String {
        case controlFrame = "control_frame"
        case editorPane = "editor_pane"
        case anchor = "anchor"
    }

    private struct AnchorResolution {
        let point: NSPoint
        let source: AnchorSource
        let fallbackReason: AnchorFallbackReason?
        let highlightRect: CGRect?
        let focusedRole: String?
        let focusedSubrole: String?
        let diagnostics: String
    }

    private struct FrameAnchorInfo {
        let point: NSPoint
        let rect: CGRect
    }

    private struct CaretAnchorInfo {
        let point: NSPoint
        let rect: CGRect
    }

    private struct AnchorSnapshot: Equatable {
        let source: AnchorSource
        let placement: OverlayPlacement
        let fallbackReason: AnchorFallbackReason?
        let anchorX: Int
        let anchorY: Int
        let screenName: String
        let highlightX: Int?
        let highlightY: Int?
        let highlightW: Int?
        let highlightH: Int?
    }

    private var panels: [FloatingPanel<AnyView>] = []
    private var isPresented = false
    private var currentAppState: AppState?
    private let overlaySize = CGSize(width: 540, height: 210)
    private let followIntervalMs: Int = 60
    private var dismissTask: Task<Void, Never>?
    private var followTask: Task<Void, Never>?
    private var lastAnchorSnapshot: AnchorSnapshot?
    private var focusHighlightPanel: FocusHighlightPanel?
    private var connectorPanel: OverlayConnectorPanel?
    private var lastStableAnchorPoint: NSPoint?
    private var lastStableScreen: NSScreen?

    func show(appState: AppState) {
        // Cancel any pending dismiss from a previous recording.
        dismissTask?.cancel()
        dismissTask = nil

        currentAppState = appState
        appState.overlayDismiss = .none

        // Close stale panels.
        if !panels.isEmpty {
            for panel in panels { panel.close() }
            panels.removeAll()
        }

        let binding = Binding<Bool>(
            get: { [weak self] in
                self?.isPresented ?? false
            },
            set: { [weak self] newValue in
                guard let self else { return }
                self.isPresented = newValue
                if !newValue {
                    self.panels.removeAll()
                }
            }
        )

        let contentRect = NSRect(origin: .zero, size: overlaySize)
        let panel = FloatingPanel(
            view: {
                AnyView(
                    RecordingOverlayView(appState: appState)
                )
            },
            contentRect: contentRect,
            isPresented: binding
        )
        panel.alphaValue = 1.0
        panel.orderFrontRegardless()
        panels.append(panel)

        isPresented = true
        refreshOverlayAnchor(logOnChangeOnly: false, event: "show")
        startFollowLoop()
    }

    func hide() {
        dismissTask?.cancel()
        dismissTask = nil
        followTask?.cancel()
        followTask = nil
        lastAnchorSnapshot = nil
        lastStableAnchorPoint = nil
        lastStableScreen = nil
        connectorPanel?.orderOut(nil)
        connectorPanel?.close()
        connectorPanel = nil
        focusHighlightPanel?.orderOut(nil)
        focusHighlightPanel?.close()
        focusHighlightPanel = nil
        isPresented = false
        for panel in panels { panel.close() }
        panels.removeAll()
        currentAppState?.overlayDismiss = .none
        currentAppState?.overlayControlFrameMode = false
        currentAppState = nil
    }

    /// Trigger dismiss animation and return immediately (non-blocking).
    func hideWithResult(_ result: OverlayResult) {
        guard result != .none, let appState = currentAppState else {
            hide()
            return
        }

        // Tell the SwiftUI views to animate the dismiss (all panels share appState).
        appState.overlayDismiss = result

        // Schedule cleanup after the animation plays.
        dismissTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(280))
            guard !Task.isCancelled else { return }
            self.hide()
        }
    }

    private func startFollowLoop() {
        followTask?.cancel()
        followTask = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(self.followIntervalMs))
                guard !Task.isCancelled else { return }
                guard self.isPresented, !self.panels.isEmpty else { continue }
                self.refreshOverlayAnchor(logOnChangeOnly: true, event: "update")
            }
        }
    }

    private func refreshOverlayAnchor(logOnChangeOnly: Bool, event: String) {
        guard !panels.isEmpty else { return }

        guard currentAppState?.phase == .recording else {
            currentAppState?.overlayTetherOutOfApp = false
            updateFocusHighlight(with: nil)
            connectorPanel?.orderOut(nil)
            return
        }

        if let lockedBundle = currentAppState?.overlayLockedBundleID,
           let frontBundle = NSWorkspace.shared.frontmostApplication?.bundleIdentifier,
           frontBundle != lockedBundle {
            currentAppState?.overlayTetherOutOfApp = true
            if !logOnChangeOnly {
                Self.logger.warning(
                    "[hark] overlay-anchor event=\(event, privacy: .public) source=locked reason=app_mismatch locked_bundle='\(lockedBundle, privacy: .public)' front_bundle='\(frontBundle, privacy: .public)'"
                )
            }
            // Move overlay to top-center of the screen
            if let screen = NSScreen.main {
                let screenFrame = screen.visibleFrame
                let x = screenFrame.midX - overlaySize.width / 2
                let y = screenFrame.maxY - overlaySize.height - 20
                for panel in panels {
                    panel.setFrameOrigin(NSPoint(x: x, y: y))
                }
            }
            updateFocusHighlight(with: nil)
            connectorPanel?.orderOut(nil)
            return
        }
        currentAppState?.overlayTetherOutOfApp = false

        let resolution = Self.resolveAnchorPoint()
        var anchorPoint = resolution.point
        let targetScreen: NSScreen?

        if resolution.source == .mouse {
            if let stablePoint = lastStableAnchorPoint {
                anchorPoint = stablePoint
                targetScreen = lastStableScreen ?? Self.screen(containing: stablePoint)
            } else {
                guard let fallbackScreen = NSScreen.main else { return }
                anchorPoint = Self.defaultFallbackAnchor(on: fallbackScreen)
                targetScreen = fallbackScreen
            }
        } else {
            targetScreen = Self.screen(containing: anchorPoint)
            lastStableAnchorPoint = anchorPoint
            lastStableScreen = targetScreen
        }

        guard let targetScreen else {
            return
        }

        let frontBundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        let placement = Self.overlayPlacement(for: resolution, on: targetScreen, bundleID: frontBundleID)
        var footerAbove = false
        if placement == .controlFrame, let targetRect = resolution.highlightRect {
            footerAbove = targetRect.midY <= targetScreen.visibleFrame.midY
        }
        currentAppState?.overlayFooterAbove = footerAbove
        currentAppState?.overlayControlFrameMode = (placement == .controlFrame)

        let targetRect = resolution.highlightRect

        for panel in panels {
            if let targetRect, placement == .controlFrame {
                panel.positionOverlayOnInputRect(
                    targetRect,
                    footerAbove: footerAbove,
                    on: targetScreen
                )
            } else if let targetRect, placement == .editorPane {
                panel.positionNearEditorRect(targetRect, on: targetScreen)
            } else {
                panel.positionNearCursor(anchorPoint, on: targetScreen)
            }
        }
        // Disable highlight/connector visuals entirely; overlay itself is the only indicator.
        let highlightRect: CGRect? = nil
        updateFocusHighlight(with: nil)

        guard let panel = panels.first else { return }
        updateConnector(bubbleFrame: panel.frame, targetRect: nil)

        let snapshot = AnchorSnapshot(
            source: resolution.source,
            placement: placement,
            fallbackReason: resolution.fallbackReason,
            anchorX: Int(anchorPoint.x.rounded()),
            anchorY: Int(anchorPoint.y.rounded()),
            screenName: targetScreen.localizedName,
            highlightX: highlightRect.map { Int($0.origin.x.rounded()) },
            highlightY: highlightRect.map { Int($0.origin.y.rounded()) },
            highlightW: highlightRect.map { Int($0.width.rounded()) },
            highlightH: highlightRect.map { Int($0.height.rounded()) }
        )
        let changed = snapshot != lastAnchorSnapshot
        if !logOnChangeOnly || changed {
            logAnchor(
                event: event,
                resolution: resolution,
                placement: placement,
                anchorPoint: anchorPoint,
                panelFrame: panel.frame,
                targetScreen: targetScreen
            )
        }
        lastAnchorSnapshot = snapshot
    }

    private func updateConnector(bubbleFrame: CGRect, targetRect: CGRect?) {
        guard let targetRect, targetRect.width > 8, targetRect.height > 8 else {
            connectorPanel?.orderOut(nil)
            return
        }
        if bubbleFrame.intersects(targetRect.insetBy(dx: -2, dy: -2)) {
            connectorPanel?.orderOut(nil)
            return
        }
        if connectorPanel == nil {
            connectorPanel = OverlayConnectorPanel()
        }

        let targetCenter = CGPoint(x: targetRect.midX, y: targetRect.midY)
        let start: CGPoint
        let end: CGPoint

        if bubbleFrame.minY >= targetRect.maxY {
            start = CGPoint(
                x: min(max(targetCenter.x, bubbleFrame.minX + 24), bubbleFrame.maxX - 24),
                y: bubbleFrame.minY
            )
            end = CGPoint(x: targetCenter.x, y: targetRect.maxY)
        } else if bubbleFrame.maxY <= targetRect.minY {
            start = CGPoint(
                x: min(max(targetCenter.x, bubbleFrame.minX + 24), bubbleFrame.maxX - 24),
                y: bubbleFrame.maxY
            )
            end = CGPoint(x: targetCenter.x, y: targetRect.minY)
        } else if bubbleFrame.midX < targetCenter.x {
            start = CGPoint(
                x: bubbleFrame.maxX,
                y: min(max(targetCenter.y, bubbleFrame.minY + 20), bubbleFrame.maxY - 20)
            )
            end = CGPoint(x: targetRect.minX, y: targetCenter.y)
        } else {
            start = CGPoint(
                x: bubbleFrame.minX,
                y: min(max(targetCenter.y, bubbleFrame.minY + 20), bubbleFrame.maxY - 20)
            )
            end = CGPoint(x: targetRect.maxX, y: targetCenter.y)
        }

        connectorPanel?.setConnector(from: start, to: end)
    }

    private static func screen(containing point: NSPoint) -> NSScreen? {
        NSScreen.screens.first(where: { screen in
            screen.frame.contains(point)
        }) ?? NSScreen.main
    }

    private static func defaultFallbackAnchor(on screen: NSScreen) -> NSPoint {
        let visible = screen.visibleFrame
        return NSPoint(x: visible.midX, y: visible.minY + 120)
    }

    private func updateFocusHighlight(with rect: CGRect?) {
        guard let rect,
              rect.width > 8,
              rect.height > 8 else {
            focusHighlightPanel?.orderOut(nil)
            return
        }
        if focusHighlightPanel == nil {
            focusHighlightPanel = FocusHighlightPanel()
        }
        focusHighlightPanel?.setHighlightFrame(rect)
    }

    private func logAnchor(
        event: String,
        resolution: AnchorResolution,
        placement: OverlayPlacement,
        anchorPoint: NSPoint,
        panelFrame: NSRect,
        targetScreen: NSScreen
    ) {
        let appName = NSWorkspace.shared.frontmostApplication?.localizedName ?? "unknown"
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
        let source = resolution.source.rawValue
        let placementValue = placement.rawValue
        let reason = resolution.fallbackReason?.rawValue ?? "none"
        let highlight = resolution.highlightRect.map {
            "x=\(Int($0.origin.x.rounded())) y=\(Int($0.origin.y.rounded())) w=\(Int($0.width.rounded())) h=\(Int($0.height.rounded()))"
        } ?? "none"
        Self.logger.warning(
            "[hark] overlay-anchor event=\(event, privacy: .public) source=\(source, privacy: .public) placement=\(placementValue, privacy: .public) reason=\(reason, privacy: .public) app='\(appName, privacy: .public)' bundle='\(bundleID, privacy: .public)' anchor_x=\(Int(anchorPoint.x.rounded())) anchor_y=\(Int(anchorPoint.y.rounded())) panel_x=\(Int(panelFrame.origin.x.rounded())) panel_y=\(Int(panelFrame.origin.y.rounded())) panel_w=\(Int(panelFrame.width.rounded())) panel_h=\(Int(panelFrame.height.rounded())) highlight='\(highlight, privacy: .public)' screen='\(targetScreen.localizedName, privacy: .public)' details='\(resolution.diagnostics, privacy: .public)'"
        )
    }

    private static func resolveAnchorPoint() -> AnchorResolution {
        var details: [String] = []

        guard AXIsProcessTrusted() else {
            return AnchorResolution(
                point: NSEvent.mouseLocation,
                source: .mouse,
                fallbackReason: .axNotTrusted,
                highlightRect: nil,
                focusedRole: nil,
                focusedSubrole: nil,
                diagnostics: "ax_trusted=false"
            )
        }

        let systemWide = AXUIElementCreateSystemWide()
        var focusedRef: AnyObject?
        let focusedStatus = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedRef
        )
        details.append("focused_status=\(axErrorName(focusedStatus))")
        guard focusedStatus == .success,
        let focusedRef,
        CFGetTypeID(focusedRef) == AXUIElementGetTypeID()
        else {
            return AnchorResolution(
                point: NSEvent.mouseLocation,
                source: .mouse,
                fallbackReason: .noFocusedElement,
                highlightRect: nil,
                focusedRole: nil,
                focusedSubrole: nil,
                diagnostics: details.joined(separator: " ")
            )
        }

        let element = unsafeBitCast(focusedRef, to: AXUIElement.self)
        let focusedRole = stringAttribute(kAXRoleAttribute as CFString, on: element)
        let focusedSubrole = stringAttribute(kAXSubroleAttribute as CFString, on: element)
        details.append("focused_role=\(focusedRole ?? "none")")
        details.append("focused_subrole=\(focusedSubrole ?? "none")")
        let elementFrameInfo = elementFrameAnchorInfo(for: element, details: &details, prefix: "element")
        let windowFrameInfo = focusedWindowAnchorInfo(from: systemWide, details: &details)

        if let focusedWindow = focusedWindow(from: systemWide) {
            details.append("window_role=\(stringAttribute(kAXRoleAttribute as CFString, on: focusedWindow) ?? "none")")
            details.append("window_subrole=\(stringAttribute(kAXSubroleAttribute as CFString, on: focusedWindow) ?? "none")")
            if let title = stringAttribute(kAXTitleAttribute as CFString, on: focusedWindow), !title.isEmpty {
                details.append("window_title=\(sanitizeLogValue(title))")
            }
        }

        if let caretInfo = caretAnchorInfo(
            for: element,
            elementFrame: elementFrameInfo?.rect,
            windowFrame: windowFrameInfo?.rect,
            details: &details
        ) {
            return AnchorResolution(
                point: caretInfo.point,
                source: .caret,
                fallbackReason: nil,
                highlightRect: elementFrameInfo?.rect ?? caretHighlightRect(for: caretInfo.rect),
                focusedRole: focusedRole,
                focusedSubrole: focusedSubrole,
                diagnostics: details.joined(separator: " ")
            )
        }

        if let elementFrameInfo {
            return AnchorResolution(
                point: elementFrameInfo.point,
                source: .focusedElementFrame,
                fallbackReason: nil,
                highlightRect: elementFrameInfo.rect,
                focusedRole: focusedRole,
                focusedSubrole: focusedSubrole,
                diagnostics: details.joined(separator: " ")
            )
        }

        if let windowFrameInfo {
            return AnchorResolution(
                point: windowFrameInfo.point,
                source: .focusedWindowFrame,
                fallbackReason: nil,
                highlightRect: windowFrameInfo.rect,
                focusedRole: focusedRole,
                focusedSubrole: focusedSubrole,
                diagnostics: details.joined(separator: " ")
            )
        }

        return AnchorResolution(
            point: NSEvent.mouseLocation,
            source: .mouse,
            fallbackReason: .noAccessibleGeometry,
            highlightRect: nil,
            focusedRole: focusedRole,
            focusedSubrole: focusedSubrole,
            diagnostics: details.joined(separator: " ")
        )
    }

    private static func overlayPlacement(
        for resolution: AnchorResolution,
        on screen: NSScreen,
        bundleID: String?
    ) -> OverlayPlacement {
        guard let target = resolution.highlightRect else {
            return .anchor
        }
        if isLikelyEditorPane(
            rect: target,
            source: resolution.source,
            focusedRole: resolution.focusedRole,
            focusedSubrole: resolution.focusedSubrole,
            screen: screen,
            bundleID: bundleID
        ) {
            return .editorPane
        }
        return .controlFrame
    }

    private static func isLikelyEditorPane(
        rect: CGRect,
        source: AnchorSource,
        focusedRole: String?,
        focusedSubrole: String?,
        screen: NSScreen,
        bundleID: String?
    ) -> Bool {
        let role = (focusedRole ?? "").lowercased()
        let subrole = (focusedSubrole ?? "").lowercased()
        let visible = screen.visibleFrame
        let bundle = (bundleID ?? "").lowercased()

        // Terminal apps should never use control-frame overlay placement.
        // Their accessibility geometry often points at transient line/row frames.
        let knownTerminalBundles: Set<String> = [
            "com.apple.terminal",
            "com.googlecode.iterm2",
            "com.mitchellh.ghostty",
            "org.alacritty",
            "com.github.wez.wezterm",
            "net.kovidgoyal.kitty",
        ]
        if knownTerminalBundles.contains(bundle) {
            return true
        }

        if source == .focusedWindowFrame {
            return true
        }

        let paneLikeRole =
            role == "axtextarea" ||
            role == "axwebarea" ||
            role == "axscrollarea" ||
            subrole.contains("terminal")
        if !paneLikeRole {
            return false
        }

        if rect.height >= 180 {
            return true
        }
        if rect.height >= visible.height * 0.35 {
            return true
        }
        if rect.width >= visible.width * 0.7 && rect.height >= 120 {
            return true
        }
        return false
    }

    private static func caretAnchorInfo(
        for element: AXUIElement,
        elementFrame: CGRect?,
        windowFrame: CGRect?,
        details: inout [String]
    ) -> CaretAnchorInfo? {
        var rangeRef: AnyObject?
        let rangeStatus = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &rangeRef
        )
        details.append("selected_range_status=\(axErrorName(rangeStatus))")
        guard rangeStatus == .success,
        let rangeRef,
        CFGetTypeID(rangeRef) == AXValueGetTypeID()
        else {
            if let rangeRef {
                details.append("selected_range_type=\(cfTypeName(rangeRef))")
            }
            return nil
        }

        let rangeValue = unsafeBitCast(rangeRef, to: AXValue.self)
        let rangeType = AXValueGetType(rangeValue)
        details.append("selected_range_type=\(axValueTypeName(rangeType))")
        guard rangeType == .cfRange else {
            return nil
        }

        var boundsRef: AnyObject?
        let status = AXUIElementCopyParameterizedAttributeValue(
            element,
            kAXBoundsForRangeParameterizedAttribute as CFString,
            rangeValue,
            &boundsRef
        )
        details.append("bounds_for_range_status=\(axErrorName(status))")
        guard status == .success,
              let boundsRef,
              CFGetTypeID(boundsRef) == AXValueGetTypeID()
        else {
            if let boundsRef {
                details.append("bounds_for_range_type=\(cfTypeName(boundsRef))")
            }
            return nil
        }

        let boundsValue = unsafeBitCast(boundsRef, to: AXValue.self)
        let boundsType = AXValueGetType(boundsValue)
        details.append("bounds_for_range_type=\(axValueTypeName(boundsType))")
        guard boundsType == .cgRect else {
            return nil
        }

        var rawRect = CGRect.zero
        guard AXValueGetValue(boundsValue, .cgRect, &rawRect) else {
            return nil
        }

        details.append("caret_rect_raw_x=\(Int(rawRect.origin.x.rounded()))")
        details.append("caret_rect_raw_y=\(Int(rawRect.origin.y.rounded()))")
        details.append("caret_rect_raw_w=\(Int(rawRect.width.rounded()))")
        details.append("caret_rect_raw_h=\(Int(rawRect.height.rounded()))")

        if rawRect.width <= 0 && rawRect.height <= 0 {
            details.append("caret_rect_rejected=zero_area")
            return nil
        }

        let rect = convertAXRectToScreenSpace(rawRect, details: &details, prefix: "caret")
        let normalizedRect = normalizeCaretRect(
            rect,
            elementFrame: elementFrame,
            windowFrame: windowFrame,
            details: &details
        )
        details.append("caret_rect_norm_x=\(Int(normalizedRect.origin.x.rounded()))")
        details.append("caret_rect_norm_y=\(Int(normalizedRect.origin.y.rounded()))")
        details.append("caret_rect_norm_w=\(Int(normalizedRect.width.rounded()))")
        details.append("caret_rect_norm_h=\(Int(normalizedRect.height.rounded()))")

        let point = NSPoint(x: normalizedRect.midX, y: normalizedRect.maxY)
        if let elementFrame,
           !contains(point: point, in: elementFrame, tolerance: 14) {
            details.append("caret_rect_rejected=outside_element_frame")
            return nil
        }
        if let windowFrame,
           !contains(point: point, in: windowFrame, tolerance: 24) {
            details.append("caret_rect_rejected=outside_window_frame")
            return nil
        }

        return CaretAnchorInfo(point: point, rect: normalizedRect)
    }

    private static func caretHighlightRect(for caretRect: CGRect) -> CGRect {
        let width = max(caretRect.width, 2)
        let height = max(caretRect.height, 16)
        let visibleLineRect = CGRect(
            x: caretRect.midX - (width / 2),
            y: caretRect.origin.y,
            width: width,
            height: height
        )
        return visibleLineRect.insetBy(dx: -8, dy: -4)
    }

    private static func normalizeCaretRect(
        _ rawRect: CGRect,
        elementFrame: CGRect?,
        windowFrame: CGRect?,
        details: inout [String]
    ) -> CGRect {
        guard let elementFrame else {
            return rawRect
        }

        let rawCenter = CGPoint(x: rawRect.midX, y: rawRect.midY)
        if contains(point: rawCenter, in: elementFrame, tolerance: 6) {
            details.append("caret_rect_space=absolute")
            return rawRect
        }

        let looksElementRelative =
            rawRect.minX >= -2 &&
            rawRect.minY >= -2 &&
            rawRect.maxX <= elementFrame.width + 2 &&
            rawRect.maxY <= elementFrame.height + 2
        if looksElementRelative {
            let rebased = rawRect.offsetBy(dx: elementFrame.minX, dy: elementFrame.minY)
            if contains(point: CGPoint(x: rebased.midX, y: rebased.midY), in: elementFrame, tolerance: 6) {
                details.append("caret_rect_space=element_relative_rebased")
                return rebased
            }

            let flipped = CGRect(
                x: elementFrame.minX + rawRect.origin.x,
                y: elementFrame.maxY - rawRect.maxY,
                width: rawRect.width,
                height: rawRect.height
            )
            if contains(point: CGPoint(x: flipped.midX, y: flipped.midY), in: elementFrame, tolerance: 6) {
                details.append("caret_rect_space=element_relative_flipped")
                return flipped
            }
        }

        if let windowFrame,
           rawRect.minX >= -2,
           rawRect.minY >= -2,
           rawRect.maxX <= windowFrame.width + 2,
           rawRect.maxY <= windowFrame.height + 2 {
            let rebased = rawRect.offsetBy(dx: windowFrame.minX, dy: windowFrame.minY)
            if contains(point: CGPoint(x: rebased.midX, y: rebased.midY), in: windowFrame, tolerance: 10) {
                details.append("caret_rect_space=window_relative_rebased")
                return rebased
            }
        }

        details.append("caret_rect_space=absolute_unverified")
        return rawRect
    }

    private static func elementFrameAnchorInfo(
        for element: AXUIElement,
        details: inout [String],
        prefix: String
    ) -> FrameAnchorInfo? {
        var frameRef: AnyObject?
        let status = AXUIElementCopyAttributeValue(
            element,
            "AXFrame" as CFString,
            &frameRef
        )
        details.append("\(prefix)_frame_status=\(axErrorName(status))")
        guard status == .success,
        let frameRef,
        CFGetTypeID(frameRef) == AXValueGetTypeID()
        else {
            if let frameRef {
                details.append("\(prefix)_frame_type=\(cfTypeName(frameRef))")
            }
            return nil
        }

        let frameValue = unsafeBitCast(frameRef, to: AXValue.self)
        let frameType = AXValueGetType(frameValue)
        details.append("\(prefix)_frame_type=\(axValueTypeName(frameType))")
        guard frameType == .cgRect else {
            return nil
        }

        var rawRect = CGRect.zero
        guard AXValueGetValue(frameValue, .cgRect, &rawRect) else {
            return nil
        }
        let rect = convertAXRectToScreenSpace(rawRect, details: &details, prefix: prefix)

        details.append("\(prefix)_frame_w=\(Int(rect.width.rounded()))")
        details.append("\(prefix)_frame_h=\(Int(rect.height.rounded()))")
        details.append("\(prefix)_frame_x=\(Int(rect.origin.x.rounded()))")
        details.append("\(prefix)_frame_y=\(Int(rect.origin.y.rounded()))")
        guard rect.width > 0 || rect.height > 0 else {
            details.append("\(prefix)_frame_zero=true")
            return nil
        }

        let point = anchorPoint(for: rect, details: &details, prefix: prefix)
        return FrameAnchorInfo(point: point, rect: rect)
    }

    private static func contains(point: CGPoint, in rect: CGRect, tolerance: CGFloat) -> Bool {
        rect.insetBy(dx: -tolerance, dy: -tolerance).contains(point)
    }

    private static func convertAXRectToScreenSpace(
        _ rect: CGRect,
        details: inout [String],
        prefix: String
    ) -> CGRect {
        guard let screen = screenForAXRect(rect) else {
            details.append("\(prefix)_rect_space=ax_unknown_no_screen")
            return rect
        }

        // AX geometry uses an origin at top-left; AppKit window geometry is bottom-left.
        let localYFromTop = rect.origin.y - screen.frame.minY
        let convertedY = screen.frame.maxY - localYFromTop - rect.height
        let convertedRect = CGRect(
            x: rect.origin.x,
            y: convertedY,
            width: rect.width,
            height: rect.height
        )
        details.append("\(prefix)_rect_space=ax_top_left_converted")
        return convertedRect
    }

    private static func screenForAXRect(_ rect: CGRect) -> NSScreen? {
        let midX = rect.midX
        if let screen = NSScreen.screens.first(where: { screen in
            screen.frame.minX <= midX && midX <= screen.frame.maxX
        }) {
            return screen
        }
        return NSScreen.main
    }

    private static func focusedWindowAnchorInfo(from systemWide: AXUIElement, details: inout [String]) -> FrameAnchorInfo? {
        guard let windowElement = focusedWindow(from: systemWide) else {
            details.append("focused_window_status=not_found")
            return nil
        }
        return elementFrameAnchorInfo(for: windowElement, details: &details, prefix: "window")
    }

    private static func anchorPoint(for rect: CGRect, details: inout [String], prefix: String) -> NSPoint {
        // For very tall editors/terminals where we only have the element frame,
        // anchoring to maxY sticks near the top of the window and feels wrong.
        let anchorY: CGFloat
        if rect.height >= 280 {
            anchorY = rect.minY + min(72, rect.height * 0.2)
            details.append("\(prefix)_anchor_hint=tall_frame_bottom_bias")
        } else {
            anchorY = rect.maxY
        }
        return NSPoint(x: rect.midX, y: anchorY)
    }

    private static func focusedWindow(from systemWide: AXUIElement) -> AXUIElement? {
        var windowRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedWindowAttribute as CFString,
            &windowRef
        ) == .success,
        let windowRef,
        CFGetTypeID(windowRef) == AXUIElementGetTypeID()
        else {
            return nil
        }

        return unsafeBitCast(windowRef, to: AXUIElement.self)
    }

    private static func stringAttribute(_ attribute: CFString, on element: AXUIElement) -> String? {
        var valueRef: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute, &valueRef) == .success,
              let valueRef
        else {
            return nil
        }
        return valueRef as? String
    }

    private static func sanitizeLogValue(_ value: String) -> String {
        value.replacingOccurrences(of: "'", with: "’")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
    }

    private static func cfTypeName(_ value: AnyObject) -> String {
        let typeID = CFGetTypeID(value)
        if typeID == AXValueGetTypeID() {
            let axValue = unsafeBitCast(value, to: AXValue.self)
            return "AXValue.\(axValueTypeName(AXValueGetType(axValue)))"
        }
        if typeID == AXUIElementGetTypeID() {
            return "AXUIElement"
        }
        if typeID == CFStringGetTypeID() {
            return "CFString"
        }
        if typeID == CFNumberGetTypeID() {
            return "CFNumber"
        }
        if typeID == CFBooleanGetTypeID() {
            return "CFBoolean"
        }
        if typeID == CFArrayGetTypeID() {
            return "CFArray"
        }
        if typeID == CFDictionaryGetTypeID() {
            return "CFDictionary"
        }
        return "CFTypeID.\(typeID)"
    }

    private static func axValueTypeName(_ type: AXValueType) -> String {
        switch type {
        case .illegal:
            return "illegal"
        case .cfRange:
            return "cfRange"
        case .cgPoint:
            return "cgPoint"
        case .cgSize:
            return "cgSize"
        case .cgRect:
            return "cgRect"
        case .axError:
            return "axError"
        @unknown default:
            return "unknown_\(type.rawValue)"
        }
    }

    private static func axErrorName(_ error: AXError) -> String {
        switch error {
        case .success:
            return "success"
        case .failure:
            return "failure"
        case .illegalArgument:
            return "illegal_argument"
        case .invalidUIElement:
            return "invalid_ui_element"
        case .invalidUIElementObserver:
            return "invalid_ui_element_observer"
        case .cannotComplete:
            return "cannot_complete"
        case .attributeUnsupported:
            return "attribute_unsupported"
        case .actionUnsupported:
            return "action_unsupported"
        case .notificationUnsupported:
            return "notification_unsupported"
        case .notImplemented:
            return "not_implemented"
        case .notificationAlreadyRegistered:
            return "notification_already_registered"
        case .notificationNotRegistered:
            return "notification_not_registered"
        case .apiDisabled:
            return "api_disabled"
        case .noValue:
            return "no_value"
        case .parameterizedAttributeUnsupported:
            return "parameterized_attribute_unsupported"
        case .notEnoughPrecision:
            return "not_enough_precision"
        @unknown default:
            return "unknown_\(error.rawValue)"
        }
    }
}
