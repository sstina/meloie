import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import App

// Horizontal dBFS bar (floor .. 0). Opaque & sharp (real-time data — never glassed).
// success / warning / error by level.
RowLayout {
    id: root
    property string label: ""
    property real dbfs: -200
    property real floorDb: -60
    property color baseColor: Theme.accent    // channel identity (IN=coral, OUT=mint); hot -> warning/error

    spacing: Theme.s2
    Layout.fillWidth: true

    Label {
        text: root.label
        color: Theme.textSecond
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fsCaption
        Layout.preferredWidth: 44
    }

    Rectangle {
        Layout.fillWidth: true
        height: 12
        radius: 6
        color: Theme.groove
        border.width: 1
        border.color: Theme.hairline

        Rectangle {
            anchors.left: parent.left
            anchors.leftMargin: 2
            anchors.verticalCenter: parent.verticalCenter
            height: parent.height - 4
            radius: height / 2
            width: Math.max(0, Math.min(1, (root.dbfs - root.floorDb) / (0 - root.floorDb)))
                   * (parent.width - 4)
            color: root.dbfs > -3 ? Theme.error : (root.dbfs > -12 ? Theme.warning : root.baseColor)
            Behavior on width { NumberAnimation { duration: 60; easing.type: Easing.OutCubic } }
        }
    }

    Label {
        text: root.dbfs <= root.floorDb ? "—" : root.dbfs.toFixed(0)
        color: Theme.textMuted
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fsCaption
        horizontalAlignment: Text.AlignRight
        Layout.preferredWidth: 36
    }
}
