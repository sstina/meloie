import QtQuick
import QtQuick.Controls
import App

// Dark SpinBox with accent focus + token +/- buttons.
SpinBox {
    id: control
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fsBody
    implicitHeight: 34

    contentItem: TextInput {
        text: control.displayText
        font: control.font
        color: control.enabled ? Theme.textPrimary : Theme.textMuted
        horizontalAlignment: Qt.AlignHCenter
        verticalAlignment: Qt.AlignVCenter
        readOnly: !control.editable
        validator: control.validator
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        selectByMouse: true
    }

    up.indicator: Rectangle {
        x: control.mirrored ? 0 : control.width - width
        height: control.height
        implicitWidth: 30
        radius: Theme.radiusSm
        color: control.up.pressed ? Theme.accentPressed
              : (control.up.hovered ? Theme.bgElevated : "transparent")
        Behavior on color { ColorAnimation { duration: control.up.pressed ? Theme.durPress : Theme.durFast } }
        Text {
            text: "+"; anchors.centerIn: parent
            color: control.up.pressed ? Theme.bgBase : Theme.textSecond
            font.pixelSize: Theme.fsTitle
            scale: control.up.pressed ? 0.82 : 1.0
            Behavior on scale { NumberAnimation {
                duration: control.up.pressed ? Theme.durPress : Theme.durRelease; easing.type: Easing.OutCubic } }
        }
    }

    down.indicator: Rectangle {
        x: control.mirrored ? control.width - width : 0
        height: control.height
        implicitWidth: 30
        radius: Theme.radiusSm
        color: control.down.pressed ? Theme.accentPressed
              : (control.down.hovered ? Theme.bgElevated : "transparent")
        Behavior on color { ColorAnimation { duration: control.down.pressed ? Theme.durPress : Theme.durFast } }
        Text {
            text: "−"; anchors.centerIn: parent
            color: control.down.pressed ? Theme.bgBase : Theme.textSecond
            font.pixelSize: Theme.fsTitle
            scale: control.down.pressed ? 0.82 : 1.0
            Behavior on scale { NumberAnimation {
                duration: control.down.pressed ? Theme.durPress : Theme.durRelease; easing.type: Easing.OutCubic } }
        }
    }

    background: Rectangle {
        implicitWidth: 124
        radius: Theme.radiusMd
        color: Theme.bgElevated
        border.width: 1
        border.color: control.activeFocus ? Theme.accent : Theme.hairline
        Behavior on border.color { ColorAnimation { duration: Theme.durFast } }
    }
}
