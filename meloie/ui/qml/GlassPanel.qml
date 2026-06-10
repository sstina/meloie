import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import App

// Flat frosted-glass card: a translucent (smoked) fill + a soft 1px border, over
// the flowing-light backdrop. No drop shadow and no top sheen (per user).
//
// Deliberately a plain Rectangle -- NO ShaderEffectSource / MultiEffect. The old
// shadow used a live:false capture re-grabbed only on resize, so a COLLAPSIBLE
// panel (高级 / 融合) painted its card from a stale, stretched snapshot while the
// text was live -> the expand/collapse distortion + card/text misalignment. A
// plain Rectangle anchored to the panel resizes in lockstep with its content, so
// that class of bug cannot occur.
Item {
    id: panelRoot
    default property alias content: inner.data
    property string title: ""
    property bool thin: false        // monitor side-panel: thinner glass so it recedes
    Layout.fillWidth: true
    implicitHeight: col.implicitHeight + 2 * Theme.s4

    Rectangle {
        id: shape
        anchors.fill: parent
        radius: Theme.glassRadius
        antialiasing: true
        color: Theme.glassEnabled ? (panelRoot.thin ? Theme.glassPanelBg : Theme.glassCard)
                                  : Theme.bgSurface
        border.width: 1
        border.color: Theme.glassEnabled ? Theme.glassBorder : Theme.hairline
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
