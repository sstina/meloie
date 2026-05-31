import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import App

// Refined 3D high-transparency glass card (Apple-ish): no hard lines — light
// diffuses in from the top edge as a soft gradient, the hairline is barely there,
// corners are large & smooth, and the card floats on a SOFT diffuse shadow.
//
// The shadow is done with ShaderEffectSource + a standalone MultiEffect (NOT
// layer.effect): layer.effect clips the blur to the card's own bounds, which made
// the shadow look like a boxy black rectangle. A standalone MultiEffect with
// autoPadding can spread the blur well beyond the card, so it reads as 氤氲/diffuse.
// The shadow is captured once (live:false) and only re-grabbed on resize / glass
// changes — the real-time content is a separate SHARP sibling, never in here.
Item {
    id: panelRoot
    default property alias content: inner.data
    property string title: ""
    Layout.fillWidth: true
    implicitHeight: col.implicitHeight + 2 * Theme.s4

    Component.onCompleted: shapeTex.scheduleUpdate()
    onWidthChanged: shapeTex.scheduleUpdate()
    onHeightChanged: shapeTex.scheduleUpdate()
    Connections {
        target: Theme
        function onGlassEnabledChanged() { shapeTex.scheduleUpdate() }
        function onGlassQualityChanged() { shapeTex.scheduleUpdate() }
    }

    // The glass shape. Glass ON -> drawn via the SES+MultiEffect below (with shadow);
    // glass OFF -> shown directly as a flat opaque panel.
    Rectangle {
        id: shape
        anchors.fill: parent
        radius: Theme.glassRadius
        antialiasing: true
        visible: !Theme.glassEnabled
        color: Theme.glassEnabled ? Theme.glassTint : Theme.bgSurface
        border.width: 1
        border.color: Theme.glassEnabled ? Theme.glassBorder : Theme.hairline

        // soft top glow — light diffusing in from the top edge as a smooth gradient
        // (replaces the old hard 1px white rim line that looked pasted-on).
        Rectangle {
            visible: Theme.glassEnabled
            anchors { left: parent.left; right: parent.right; top: parent.top; margins: 1 }
            height: parent.radius * 1.8
            radius: parent.radius - 1
            gradient: Gradient {
                GradientStop { position: 0.0;  color: Theme.glassTopGlow }
                GradientStop { position: 0.45; color: Qt.rgba(1, 1, 1, 0.02) }
                GradientStop { position: 1.0;  color: "transparent" }
            }
        }
        // soft bottom depth — a gentle dark fade up from the bottom (no hard line)
        Rectangle {
            visible: Theme.glassEnabled
            anchors { left: parent.left; right: parent.right; bottom: parent.bottom; margins: 1 }
            height: parent.radius * 1.4
            radius: parent.radius - 1
            gradient: Gradient {
                GradientStop { position: 0.0; color: "transparent" }
                GradientStop { position: 1.0; color: Theme.glassUnderShadow }
            }
        }
    }

    ShaderEffectSource {
        id: shapeTex
        anchors.fill: shape
        sourceItem: shape
        live: false            // shape is static; recaptured via scheduleUpdate() above
        visible: false
    }
    MultiEffect {
        anchors.fill: shape
        source: shapeTex
        visible: Theme.glassEnabled
        autoPaddingEnabled: true          // let the shadow spread beyond the card
        shadowEnabled: true
        shadowColor: Theme.glassShadow
        shadowBlur: 1.0
        blurMax: Theme.glassShadowBlurMax
        shadowVerticalOffset: 16
        shadowHorizontalOffset: 0
    }

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: Theme.s4
        spacing: Theme.s2

        Label {
            text: panelRoot.title
            visible: panelRoot.title.length > 0
            color: Theme.textSecond
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fsLabel
            font.weight: Theme.fwSemibold
            font.letterSpacing: 0.5
        }
        ColumnLayout { id: inner; Layout.fillWidth: true; spacing: Theme.s2 }
    }
}
