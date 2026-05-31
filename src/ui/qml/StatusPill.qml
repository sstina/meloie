import QtQuick
import QtQuick.Controls
import App

// Tinted status pill (idle/loading/loaded/running/stopping/error) with a dot.
Rectangle {
    id: root
    property string statusText: "idle"

    readonly property color statusColor: {
        switch (root.statusText) {
        case "running":  return Theme.success;
        case "loading":  return Theme.warning;
        case "stopping": return Theme.warning;
        case "loaded":   return Theme.accent;
        case "error":    return Theme.error;
        default:         return Theme.textMuted;
        }
    }

    implicitWidth: row.implicitWidth + 22
    implicitHeight: 26
    radius: height / 2
    color: Qt.rgba(statusColor.r, statusColor.g, statusColor.b, 0.18)
    border.width: 1
    border.color: Qt.rgba(statusColor.r, statusColor.g, statusColor.b, 0.5)
    Behavior on color { ColorAnimation { duration: Theme.durBase } }

    readonly property bool busy: root.statusText === "loading" || root.statusText === "stopping"

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 6
        Rectangle {
            id: dot
            width: 7; height: 7; radius: 3.5
            anchors.verticalCenter: parent.verticalCenter
            color: root.statusColor
            // breathe while busy (loading the ~30s model / stopping) so the
            // pill reads as "working", not stuck. Resets to solid when idle.
            SequentialAnimation {
                running: root.busy
                loops: Animation.Infinite
                alwaysRunToEnd: false
                NumberAnimation { target: dot; property: "opacity"; to: 0.35; duration: 550; easing.type: Easing.InOutQuad }
                NumberAnimation { target: dot; property: "opacity"; to: 1.0;  duration: 550; easing.type: Easing.InOutQuad }
                onRunningChanged: if (!running) dot.opacity = 1.0
            }
        }
        Label {
            text: root.statusText
            color: Theme.textPrimary
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fsCaption
            font.weight: Theme.fwMedium
        }
    }
}
