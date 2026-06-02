import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import App

// label + accent Slider + live value readout. Emits moved(value) during a drag.
RowLayout {
    id: root
    property string label: ""
    property real from: 0
    property real to: 1
    property real stepSize: 0.01
    property int decimals: 2
    property string suffix: ""
    property color accentColor: Theme.accent     // semantic hue (pitch=sky, formant=lilac, ...)
    property alias value: slider.value
    signal moved(real value)

    spacing: Theme.s3
    Layout.fillWidth: true

    Label {
        text: root.label
        visible: root.label.length > 0
        color: Theme.textSecond
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fsBody
        Layout.preferredWidth: 96
    }

    Slider {
        id: slider
        Layout.fillWidth: true
        from: root.from
        to: root.to
        stepSize: root.stepSize
        enabled: root.enabled
        onMoved: root.moved(value)

        background: Rectangle {
            x: slider.leftPadding
            y: slider.topPadding + slider.availableHeight / 2 - height / 2
            width: slider.availableWidth
            height: 4
            radius: 2
            color: Theme.groove
            Rectangle {
                width: slider.visualPosition * parent.width
                height: parent.height
                radius: 2
                color: slider.enabled ? root.accentColor : Theme.textMuted
            }
        }

        handle: Rectangle {
            x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
            y: slider.topPadding + slider.availableHeight / 2 - height / 2
            implicitWidth: 16
            implicitHeight: 16
            radius: 8
            color: slider.pressed ? Qt.darker(root.accentColor, 1.2) : root.accentColor
            border.width: slider.enabled ? 1 : 0
            border.color: Qt.rgba(1, 1, 1, 0.25)
            opacity: slider.enabled ? 1.0 : 0.4
            scale: slider.pressed ? 1.28 : (slider.hovered ? 1.12 : 1.0)   // grab = bigger
            Behavior on scale { NumberAnimation {
                duration: slider.pressed ? Theme.durPress : Theme.durRelease; easing.type: Easing.OutCubic } }

            // soft grab halo on press (purely cosmetic)
            Rectangle {
                anchors.centerIn: parent
                width: parent.width + 12; height: width; radius: width / 2
                color: "transparent"
                border.width: 2
                border.color: root.accentColor
                opacity: slider.pressed ? 0.35 : 0.0
                Behavior on opacity { NumberAnimation { duration: Theme.durFast } }
            }
        }
    }

    Label {
        text: slider.value.toFixed(root.decimals) + root.suffix
        color: Theme.textSecond
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fsBody
        horizontalAlignment: Text.AlignRight
        Layout.preferredWidth: 64
    }
}
