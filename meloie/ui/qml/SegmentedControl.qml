import QtQuick
import QtQuick.Controls
import App

// A multi-state segmented selector with a sliding highlight pill — the "三滑按钮"
// (a 3+ state toggle, not an on/off switch). Width is set by the parent (e.g.
// Layout.preferredWidth) so segments are equal-width with NO circular binding;
// the pill animates its x between segments (the "slide"). Generic/reusable:
// pass `options` (labels) + bind `currentIndex`, handle `activated(index)`.
Item {
    id: root

    property var options: []                 // segment labels (strings)
    property int currentIndex: 0
    property color accent: Theme.accent
    // when true, index 0 uses a neutral pill (so an "off"/"关" first segment is not
    // highlighted in accent — which would read as "on").
    property bool dimFirst: false
    property int pad: 3
    signal activated(int index)

    implicitHeight: 30
    readonly property int count: options ? options.length : 0
    readonly property real segW: count > 0 ? Math.max(0, (width - 2 * pad) / count) : 0

    // track
    Rectangle {
        anchors.fill: parent
        radius: height / 2
        color: Theme.glassField
        border.width: 1
        border.color: Theme.glassBorder
    }

    // sliding highlight pill
    Rectangle {
        id: pill
        visible: root.count > 0
        width: root.segW
        height: parent.height - 2 * root.pad
        x: root.pad + root.currentIndex * root.segW
        y: root.pad
        radius: height / 2
        color: (root.dimFirst && root.currentIndex === 0)
               ? Theme.bgElevated
               : Qt.rgba(root.accent.r, root.accent.g, root.accent.b, root.enabled ? 0.92 : 0.4)
        Behavior on x { NumberAnimation { duration: Theme.durBase; easing.type: Easing.OutCubic } }
        Behavior on color { ColorAnimation { duration: Theme.durBase } }
    }

    Row {
        anchors.fill: parent
        anchors.margins: root.pad
        Repeater {
            model: root.options
            delegate: Item {
                width: root.segW
                height: parent.height
                Text {
                    anchors.centerIn: parent
                    text: modelData
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fsLabel
                    font.weight: index === root.currentIndex ? Theme.fwSemibold : Theme.fwRegular
                    color: index === root.currentIndex
                           ? ((root.dimFirst && index === 0) ? Theme.textPrimary : Theme.bgBase)
                           : (root.enabled ? Theme.textSecond : Theme.textMuted)
                    Behavior on color { ColorAnimation { duration: Theme.durFast } }
                }
                MouseArea {
                    anchors.fill: parent
                    enabled: root.enabled
                    cursorShape: Qt.PointingHandCursor
                    onClicked: { root.currentIndex = index; root.activated(index); }
                }
            }
        }
    }

    opacity: enabled ? 1.0 : 0.55
    Behavior on opacity { NumberAnimation { duration: Theme.durFast } }
}
