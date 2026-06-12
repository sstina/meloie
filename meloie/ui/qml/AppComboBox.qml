import QtQuick
import QtQuick.Controls
import App

// Dark, accent-focused ComboBox (Basic style). Supports both plain string models
// and QVariantList/array-of-dicts models via `textRole` (used by profiles/devices).
ComboBox {
    id: control
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fsBody
    implicitHeight: 34

    scale: control.pressed ? 0.99 : 1.0
    Behavior on scale {
        NumberAnimation {
            duration: control.pressed ? Theme.durPress : Theme.durRelease
            easing.type: Easing.OutCubic
        }
    }

    function _roleText(md) {
        return (control.textRole && control.textRole.length > 0 && md !== undefined && md !== null
                && typeof md === "object")
               ? md[control.textRole]
               : md;
    }

    contentItem: Text {
        leftPadding: Theme.s3
        rightPadding: control.indicator.width + Theme.s2
        text: control.displayText
        font: control.font
        color: Theme.textPrimary
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Text {
        x: control.width - width - Theme.s3
        y: control.topPadding + (control.availableHeight - height) / 2
        text: "▾"
        font.pixelSize: 10
        color: Theme.textSecond
    }

    background: Rectangle {
        radius: Theme.radiusMd
        // translucent glass field when the glass theme is on (opaque fallback off)
        color: Theme.glassEnabled
               ? (control.pressed ? Qt.rgba(Theme.bgSurface.r, Theme.bgSurface.g, Theme.bgSurface.b, 0.65)
                                  : Theme.glassField)
               : (control.pressed ? Theme.bgSurface : Theme.bgElevated)
        border.width: 1
        border.color: (control.activeFocus || control.pressed) ? Theme.accent : Theme.hairline
        Behavior on color { ColorAnimation { duration: Theme.durFast } }
        Behavior on border.color { ColorAnimation { duration: Theme.durFast } }
    }

    delegate: ItemDelegate {
        id: deleg
        width: control.width
        required property var modelData
        required property int index
        contentItem: Text {
            text: control._roleText(deleg.modelData)
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fsBody
            color: Theme.textPrimary
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        highlighted: control.highlightedIndex === deleg.index
        background: Rectangle {
            color: deleg.highlighted
                   ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.18)
                   : "transparent"
        }
    }

    popup: Popup {
        y: control.height + 4
        width: control.width
        implicitHeight: Math.min(listView.contentHeight + 8, 280)
        padding: 4
        contentItem: ListView {
            id: listView
            clip: true
            implicitHeight: contentHeight
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            ScrollIndicator.vertical: ScrollIndicator { }
        }
        background: Rectangle {
            radius: Theme.radiusMd
            color: Theme.bgSurface
            border.width: 1
            border.color: Theme.hairline
        }
    }
}
