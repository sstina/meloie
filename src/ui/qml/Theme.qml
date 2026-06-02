pragma Singleton
import QtQuick

// Design tokens + the (writable) glass enhancement switches.
// Registered from app.py via qmlRegisterSingletonType(uri="App", name="Theme").
// Pure presentation: no backend, no audio, no contract surface.
//
// Palette = "Morning Bloom" (see DESIGN_BRIEF.md): a cold-base + warm-accent set
// where every functional dimension owns one hue (颜色=信息). The six source colors
// below are the single source of truth; the semantic aliases under them are what
// the rest of the UI references, so a re-map happens in exactly one place.
QtObject {
    id: theme

    // ---- backgrounds (60% : deep, low-saturation; brief base #0E141B) ----
    readonly property color bgBase:     "#0E141B"
    readonly property color bgSurface:  "#1A2129"
    readonly property color bgElevated: "#232D38"

    // ---- Morning Bloom source palette (six accents; the single truth source) ----
    readonly property color mint:    "#34C7A9"   // 平衡/新生 -> 运行/前进/正向
    readonly property color sky:     "#3EA9EE"   // 通透/信任 -> 变调 pitch
    readonly property color lilac:   "#B295E6"   // 温柔/创意 -> 声线/共振/融合
    readonly property color peach:   "#FF9DB0"   // 亲和/安抚 -> 预留(轻提示)
    readonly property color coral:   "#FF7A66"   // 活力/友好 -> 输入能量 IN / 停止 Stop
    readonly property color sunbeam: "#FFC24A"   // 乐观/愉悦 -> 告警 warning
    // brighter variants for small numerals on the dark glass (pure color reads muddy small)
    readonly property color mintSoft:  "#7DEBD3"
    readonly property color skySoft:   "#8CCBF4"
    readonly property color lilacSoft: "#CFBDF2"
    readonly property color coralSoft: "#FF9E8C"

    // ---- semantic aliases (颜色=信息; remap here, never at the call site) ----
    readonly property color accent:        mint        // active / forward / OUT / routing-OK
    readonly property color accentHover:   "#54D9BE"   // mint, lifted
    readonly property color accentPressed: "#26A98E"   // mint, pushed
    readonly property color pitch:     sky             // 变调维度
    readonly property color resonance: lilac           // 性别/共振 + 融合
    readonly property color input:     coral           // IN 电平 / Stop
    readonly property color ok:        mintSoft         // 正向小字 (deep-bg numerals)

    // ---- text (WCAG verified on dark; see plan) ----
    readonly property color textPrimary: "#F1F5F9"   // 16.9:1 on bgBase
    readonly property color textSecond:  "#94A3B8"   // 7.2:1  on bgBase
    readonly property color textMuted:   "#64748B"   // 3.9:1  -> non-essential labels only

    // ---- status ----
    readonly property color success: mint            // 运行/正向 (shares the mint hue)
    readonly property color warning: sunbeam         // 需注意但非错误
    readonly property color error:   "#F87171"       // critical alert (brief omits a red; kept distinct from coral=IN/Stop)

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

    // Morning Bloom glass (brief §2.3 intent): frosted card that lets the drifting
    // flowing-light show through. Implementation note: the brief's literal
    // "white-alpha" fill is rendered here as a SMOKED-glass tint (semi-transparent
    // dark). On a dark theme this composites identically to "translucent white over the
    // faint backdrop", but a semi-opaque fill (a) casts a real drop shadow and
    // (b) occludes its own shadow (no bleed-through), and (c) keeps light text
    // >=4.5:1 even when a bright blob drifts behind it. Verified via check_qml.
    readonly property color glassCard:   Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, 0.50)   // normal card
    readonly property color glassPanelBg:Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, 0.38)   // monitor side panel (thinner -> recedes)
    readonly property color glassField:  Qt.rgba(bgElevated.r, bgElevated.g, bgElevated.b, 0.55) // input / combo / small container
    readonly property color groove:      Qt.rgba(1, 1, 1, 0.12)   // slider/meter trough
    readonly property color hoverFill:   Qt.rgba(1, 1, 1, 0.10)   // hover/press wash

    readonly property color glassBorder: Qt.rgba(1, 1, 1, 0.10)   // soft hairline (no hard outline)
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
