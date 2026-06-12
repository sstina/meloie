pragma Singleton
import QtQuick

// Design tokens + the (writable) glass enhancement switches.
// Registered from app.py via qmlRegisterSingletonType(uri="App", name="Theme").
// Pure presentation: no backend, no audio, no contract surface.
//
// Palette = "Morning Bloom" (see DESIGN_BRIEF.md): a cold-base + warm-accent set
// where every functional dimension owns one hue (颜色=信息). The five source colors
// below are the single source of truth; the semantic aliases under them are what
// the rest of the UI references, so a re-map happens in exactly one place.
QtObject {
    id: theme

    // ---- backgrounds (60% : deep, low-saturation; brief base #0E141B) ----
    readonly property color bgBase:     "#0E141B"
    readonly property color bgSurface:  "#1A2129"
    readonly property color bgElevated: "#232D38"

    // ---- Morning Bloom source palette (five accents; the single truth source) ----
    readonly property color mint:    "#34C7A9"   // 平衡/新生 -> 运行/前进/正向
    readonly property color sky:     "#3EA9EE"   // 通透/信任 -> 变调 pitch
    readonly property color lilac:   "#B295E6"   // 温柔/创意 -> 声线/共振/融合
    readonly property color coral:   "#FF7A66"   // 活力/友好 -> 输入能量 IN / 停止 Stop
    readonly property color sunbeam: "#FFC24A"   // 乐观/愉悦 -> 告警 warning

    // ---- semantic aliases (颜色=信息; remap here, never at the call site) ----
    readonly property color accent:        mint        // active / forward / OUT / routing-OK
    readonly property color accentPressed: "#26A98E"   // mint, pushed
    readonly property color pitch:     sky             // 变调维度
    readonly property color resonance: lilac           // 性别/共振 + 融合
    readonly property color input:     coral           // IN 电平 / Stop

    // ---- text (WCAG gated on the REAL composited glass via real_shoot --probe) ----
    readonly property color textPrimary: "#F1F5F9"
    // textSecond is vibrancy-adjusted (2026-06 glass rework): on the brightened
    // see-through cards the old #94A3B8 maxes out ~3.8:1 on the worst reachable
    // frame; secondary text on glass must be lighter (same move Apple makes for
    // secondaryLabel over vibrancy). Hierarchy to textPrimary is preserved.
    readonly property color textSecond:  "#BCC9D6"
    readonly property color textMuted:   "#64748B"   // non-essential labels only (by design)

    // ---- status ----
    readonly property color success: mint            // 运行/正向 (shares the mint hue)
    readonly property color warning: sunbeam         // 需注意但非错误
    readonly property color error:   "#F87171"       // critical alert (brief omits a red; kept distinct from coral=IN/Stop)

    // ---- radius ----
    readonly property int radiusSm:   6
    readonly property int radiusMd:   10

    // ---- spacing ----
    readonly property int s1: 4
    readonly property int s2: 8
    readonly property int s3: 12
    readonly property int s4: 16
    readonly property int s5: 20      // glass-card interior padding (airier, premium)

    // ---- typography ----
    readonly property string fontFamily: "Segoe UI Variable Text"
    readonly property int fsCaption: 11
    readonly property int fsLabel:   12
    readonly property int fsBody:    13
    readonly property int fsTitle:   15
    readonly property int fwRegular:  Font.Normal     // 400
    readonly property int fwMedium:   Font.Medium     // 500
    readonly property int fwSemibold: Font.DemiBold   // 600

    // ---- strokes ----
    readonly property color hairline: Qt.rgba(1, 1, 1, 0.06)

    // ---- animation ----
    readonly property int durFast: 120
    readonly property int durBase: 160
    // interaction feedback (Miller/Nielsen: press <100ms reads as "direct
    // manipulation"; consistency > speed). Asymmetric: snap down on press,
    // ease back on release. Used identically by every pressable control.
    readonly property int durPress: 90        // press-down (kept < 100ms)
    readonly property int durRelease: 160     // release settle (~150ms)
    readonly property real pressScale: 0.97   // "pushed in" amount for buttons

    // ---- glass (the toggleable enhancement layer) ----
    property bool glassEnabled: true
    property string glassQuality: "high"            // "high" | "medium" | "low"

    // High-transparency smoked glass (2026-06): the card is a LIVE vibrancy
    // sample of the flowing-light backdrop (saturated + lifted, quality "high"
    // only) under a translucent smoked tint that thickens toward the bottom
    // (glass depth), a soft top reflection wash, a lit top edge, and an
    // analytic RectangularShadow. Every alpha below is GATE-TUNED: the real-GPU
    // probe (tools/real_shoot.py --probe) holds text >=4.5:1 on the worst
    // reachable composited frame — tune there, not by eye alone.
    readonly property bool vibrancy: glassEnabled && glassQuality === "high"
    readonly property color glassCardTop:    Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, 0.42)
    readonly property color glassCardBottom: Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, 0.54)
    readonly property color glassThinTop:    Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, 0.40)
    readonly property color glassThinBottom: Qt.rgba(bgSurface.r, bgSurface.g, bgSurface.b, 0.52)
    readonly property color glassField:  Qt.rgba(bgElevated.r, bgElevated.g, bgElevated.b, 0.55) // input / combo / small container
    readonly property color groove:      Qt.rgba(1, 1, 1, 0.12)   // slider/meter trough

    readonly property color glassBorder: Qt.rgba(1, 1, 1, 0.09)   // soft hairline (no hard outline)
    readonly property color glassTopGlow: Qt.rgba(1, 1, 1, 0.22)  // 1px lit rim where light meets the top edge
    readonly property color glassSheen:  Qt.rgba(1, 1, 1, 0.05)   // soft reflection wash peak (top ~35% gradient)
    readonly property color glassShadow: Qt.rgba(0, 0, 0, 0.38)   // RectangularShadow color
    readonly property real glassVibSaturation: 0.7                // backdrop chroma x1.7 through the glass
    readonly property real glassVibBrightness: 0.03               // backdrop lift through the glass (gate-critical)
    readonly property int glassRadius: 20                         // larger, Apple-smooth corners
    // background flowing-light blur strength (quality-stepped)
    readonly property int glassBlurMax:
        glassQuality === "high" ? 72 : (glassQuality === "medium" ? 48 : 28)
    // scroll-edge dissolve (Main.qml): ONE token for three aligned things — the right
    // column's Start-button bottom inset, the left column's bottom reserve, and the
    // mask's fade-band height. Set = s4 (the body side margin) so the Start button is
    // symmetrically inset (bottom gap == right gap) and the dissolve fills exactly that
    // bottom strip; a fully-scrolled last card then rests flush on the Start line.
    readonly property int glassEdgeBand: 16
}
