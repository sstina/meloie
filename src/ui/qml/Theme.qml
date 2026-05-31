pragma Singleton
import QtQuick

// Design tokens + the (writable) glass enhancement switches.
// Registered from app.py via qmlRegisterSingletonType(uri="App", name="Theme").
// Pure presentation: no backend, no audio, no contract surface.
QtObject {
    id: theme

    // ---- backgrounds (60% : deep, low-saturation) ----
    readonly property color bgBase:     "#0F1419"
    readonly property color bgSurface:  "#1A2129"
    readonly property color bgElevated: "#232D38"

    // ---- accent (10% : interaction / active / status only) ----
    readonly property color accent:        "#2DD4BF"
    readonly property color accentHover:   "#5EEAD4"
    readonly property color accentPressed: "#14B8A6"

    // ---- text (WCAG verified on dark; see plan) ----
    readonly property color textPrimary: "#F1F5F9"   // 16.9:1 on bgBase
    readonly property color textSecond:  "#94A3B8"   // 7.2:1  on bgBase
    readonly property color textMuted:   "#64748B"   // 3.9:1  -> non-essential labels only

    // ---- status ----
    readonly property color success: "#34D399"
    readonly property color warning: "#FBBF24"
    readonly property color error:   "#F87171"

    // ---- radius ----
    readonly property int radiusSm:   6
    readonly property int radiusMd:   10
    readonly property int radiusLg:   14
    readonly property int radiusPill: 999

    // ---- spacing ----
    readonly property int s1: 4
    readonly property int s2: 8
    readonly property int s3: 12
    readonly property int s4: 16
    readonly property int s5: 24
    readonly property int s6: 32

    // ---- typography ----
    readonly property string fontFamily: "Segoe UI Variable Text"
    readonly property int fsCaption: 11
    readonly property int fsLabel:   12
    readonly property int fsBody:    13
    readonly property int fsTitle:   15
    readonly property int fsH:       18
    readonly property int fwRegular:  Font.Normal     // 400
    readonly property int fwMedium:   Font.Medium     // 500
    readonly property int fwSemibold: Font.DemiBold   // 600

    // ---- strokes / shadow ----
    readonly property color glassHighlight: Qt.rgba(1, 1, 1, 0.08)
    readonly property color hairline:       Qt.rgba(1, 1, 1, 0.06)
    readonly property color shadowColor:    Qt.rgba(0, 0, 0, 0.35)

    // ---- animation ----
    readonly property int durFast: 120
    readonly property int durBase: 160
    readonly property int durSlow: 220
    // interaction feedback (Miller/Nielsen: press <100ms reads as "direct
    // manipulation"; consistency > speed). Asymmetric: snap down on press,
    // ease back on release. Used identically by every pressable control.
    readonly property int durPress: 90        // press-down (kept < 100ms)
    readonly property int durRelease: 160     // release settle (~150ms)
    readonly property real pressScale: 0.97   // "pushed in" amount for buttons

    // ---- glass (the toggleable enhancement layer) ----
    property bool glassEnabled: true
    property string glassQuality: "high"            // "high" | "medium" | "low"
    // High-transparency frosted glass. 0.62 lets more of the blurred backdrop show
    // through (the "高透" look) while staying dark enough for text >=4.5:1 — the
    // exact value is tuned against .tmp/check_qml.py's composited-pixel probe.
    readonly property real glassTintAlpha: 0.62
    readonly property color glassTint: Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, glassTintAlpha)
    readonly property color glassBorder: Qt.rgba(1, 1, 1, 0.09)   // soft hairline (no hard outline)
    readonly property color glassTopGlow: Qt.rgba(1, 1, 1, 0.13)  // soft top inner glow (gradient, never a line)
    readonly property color glassUnderShadow: Qt.rgba(0, 0, 0, 0.16)  // gentle bottom depth fade
    readonly property color glassShadow: Qt.rgba(0, 0, 0, 0.55)   // elevation shadow (× translucent source alpha)
    readonly property int glassRadius: 18                         // larger, Apple-smooth corners
    readonly property int glassBlurMax:
        glassQuality === "high" ? 72 : (glassQuality === "medium" ? 48 : 28)
    readonly property real glassDownsample:
        glassQuality === "high" ? 0.6 : (glassQuality === "medium" ? 0.45 : 0.33)
    // shadow blur — generous so the elevation reads as soft/氤氲, not a boxy edge
    readonly property int glassShadowBlurMax:
        glassQuality === "high" ? 64 : (glassQuality === "medium" ? 44 : 24)
}
