import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import App

// High-transparency smoked-glass card (2026-06 rework). Layered bottom -> top:
//
//   1. RectangularShadow        analytic ambient shadow (no texture capture)
//   2. vibrancy crop  (high)    LIVE ShaderEffectSource of the flowing-light
//                               backdrop, saturated + lifted, rounded-masked —
//                               the "looking THROUGH frosted glass" read
//   3. smoked tint              translucent gradient (thicker toward bottom)
//   4. reflection wash          faint white gradient over the top ~35%
//   5. lit top edge             1px horizontal-gradient rim inside the radius
//   6. content                  sharp, never inside the vibrancy layer
//
// Staleness-safety: the old (removed) glass used ShaderEffectSource(live:false)
// re-captured only on resize — collapsible panels then painted from stale
// stretched snapshots (the expand/collapse distortion bug). Here the SES is
// live:true over a continuously-animating source: a stale frame is structurally
// impossible; a missed geometry update could only mis-position the crop, and
// the per-frame FrameAnimation sync below removes that too.
//
// Quality ladder: vibrancy only at glassQuality "high"; medium/low keep the
// tint/shadow/sheen "faux glass"; glassEnabled off -> flat opaque bgSurface.
Item {
    id: panelRoot
    default property alias content: inner.data
    property string title: ""
    property bool thin: false        // monitor side-panel: thinner glass so it recedes
    // the flowing-light backdrop to sample through the glass (Main.qml passes
    // the AppBackground instance); null -> vibrancy off (e.g. tests, previews).
    property Item backdrop: null
    Layout.fillWidth: true
    implicitHeight: col.implicitHeight + 2 * Theme.s5

    readonly property bool vib: Theme.vibrancy && backdrop !== null
                                && width > 0 && height > 0

    // ---- 1. ambient shadow (analytic rounded-rect; sits under the glass, so a
    // sliver of its body shows through the translucent tint — deliberate, it
    // deepens the card center and helps the text-contrast gate).
    RectangularShadow {
        anchors.fill: parent
        visible: Theme.glassEnabled
        radius: Theme.glassRadius
        blur: 26
        spread: 0
        offset: Qt.point(0, 6)
        color: Theme.glassShadow
        cached: true
    }

    // ---- 2. vibrancy: what the backdrop looks like THROUGH the glass ----
    Item {
        id: vibLayer
        anchors.fill: parent
        visible: panelRoot.vib
        layer.enabled: panelRoot.vib
        layer.smooth: true
        layer.effect: MultiEffect {
            autoPaddingEnabled: false          // padding would misalign the mask
            maskEnabled: true
            maskSource: cardMask
            saturation: Theme.glassVibSaturation
            brightness: Theme.glassVibBrightness
        }
        ShaderEffectSource {
            id: crop
            anchors.fill: parent
            visible: panelRoot.vib
            live: true                         // staleness structurally impossible
            smooth: true                       // SES default is false -> scroll shimmer
            sourceItem: panelRoot.vib ? panelRoot.backdrop : null
        }
        // Scene-position sync. A binding can't see ancestor moves (scroll,
        // layout reflow, narrow<->wide re-stack), so while vibrancy is active
        // re-derive the crop rect each frame (one mapToItem per card per frame;
        // writes only on change). The scene animates every frame anyway.
        FrameAnimation {
            running: panelRoot.vib
            onTriggered: {
                var p = panelRoot.mapToItem(panelRoot.backdrop, 0, 0);
                var r = crop.sourceRect;
                if (r.x !== p.x || r.y !== p.y
                        || r.width !== panelRoot.width || r.height !== panelRoot.height)
                    crop.sourceRect = Qt.rect(p.x, p.y, panelRoot.width, panelRoot.height);
            }
        }
    }
    // mask provider: consumed as a texture only (same pattern as Main.qml's
    // edgeFadeMask) — gives the vibrancy layer the card's rounded corners.
    Rectangle {
        id: cardMask
        anchors.fill: parent
        visible: false
        layer.enabled: panelRoot.vib
        layer.smooth: true
        radius: Theme.glassRadius
        antialiasing: true
        color: "white"
    }

    // ---- 3. smoked tint: thin at the top, thicker at the bottom (glass depth) ----
    Rectangle {
        anchors.fill: parent
        radius: Theme.glassRadius
        antialiasing: true
        border.width: 1
        border.color: Theme.glassEnabled ? Theme.glassBorder : Theme.hairline
        gradient: Gradient {
            GradientStop {
                position: 0.0
                color: Theme.glassEnabled
                       ? (panelRoot.thin ? Theme.glassThinTop : Theme.glassCardTop)
                       : Theme.bgSurface
            }
            GradientStop {
                position: 1.0
                color: Theme.glassEnabled
                       ? (panelRoot.thin ? Theme.glassThinBottom : Theme.glassCardBottom)
                       : Theme.bgSurface
            }
        }
    }

    // ---- 4. soft reflection wash: light falling on the upper glass face.
    // NOT the old hard 1px sheen line — a faint gradient that dies out by ~35%.
    Rectangle {
        anchors.fill: parent
        radius: Theme.glassRadius
        antialiasing: true
        visible: Theme.glassEnabled
        gradient: Gradient {
            GradientStop { position: 0.0;  color: Theme.glassSheen }
            GradientStop { position: 0.35; color: "transparent" }
            GradientStop { position: 1.0;  color: "transparent" }
        }
    }

    // ---- 5. lit top edge: 1px rim inside the corner radius, brightest centre ----
    Rectangle {
        x: Theme.glassRadius
        y: 1
        width: Math.max(0, parent.width - 2 * Theme.glassRadius)
        height: 1
        visible: Theme.glassEnabled
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "transparent" }
            GradientStop { position: 0.5; color: Theme.glassTopGlow }
            GradientStop { position: 1.0; color: "transparent" }
        }
    }

    // ---- 6. content (sharp; never rendered through the vibrancy layer) ----
    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: Theme.s5
        spacing: Theme.s2

        Label {
            text: panelRoot.title
            visible: panelRoot.title.length > 0
            color: Theme.textSecond
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fsLabel
            font.weight: Theme.fwSemibold
            font.letterSpacing: 0.8
        }
        ColumnLayout { id: inner; Layout.fillWidth: true; spacing: Theme.s2 }
    }
}
