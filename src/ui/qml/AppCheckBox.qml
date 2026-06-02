import QtQuick
import QtQuick.Controls
import App

// Accent checkbox with a check glyph.
CheckBox {
    id: control
    property color accentColor: Theme.accent     // checked hue (formant/merge=lilac, else mint)
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fsBody
    spacing: Theme.s2

    indicator: Rectangle {
        implicitWidth: 18
        implicitHeight: 18
        x: control.leftPadding
        y: control.height / 2 - height / 2
        radius: Theme.radiusSm
        color: control.checked ? control.accentColor : "transparent"
        border.width: 1.5
        border.color: control.checked ? control.accentColor
                     : (control.hovered ? Theme.textSecond : Theme.textMuted)
        scale: control.down ? 0.88 : 1.0          // squish on press
        Behavior on color { ColorAnimation { duration: Theme.durFast } }
        Behavior on border.color { ColorAnimation { duration: Theme.durFast } }
        Behavior on scale { NumberAnimation {
            duration: control.down ? Theme.durPress : Theme.durRelease; easing.type: Easing.OutCubic } }

        Text {
            text: "✓"
            anchors.centerIn: parent
            color: Theme.bgBase
            font.pixelSize: 12
            font.bold: true
            visible: control.checked
        }
    }

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.enabled ? Theme.textSecond : Theme.textMuted
        opacity: control.enabled ? 1.0 : 0.5
        verticalAlignment: Text.AlignVCenter
        leftPadding: control.indicator.width + control.spacing
    }
}
