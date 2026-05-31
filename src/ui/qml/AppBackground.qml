import QtQuick
import QtQuick.Effects
import App

// Static window backdrop: a soft gradient + accent blobs, blurred ONCE.
//
// Performance contract (see plan §2): the blur source (`src`) is fully static,
// captured into `capture` with live:false, so MultiEffect runs the gaussian blur
// only at startup / resize / quality-change — never per frame. Real-time data
// (level meters, numbers) lives in the panels ABOVE this and is never sampled
// here, so it costs zero blur work. When glass is off we collapse to flat bgBase.
Item {
    id: root

    function _recapture() { capture.scheduleUpdate() }
    onWidthChanged: _recapture()
    onHeightChanged: _recapture()
    Connections {
        target: Theme
        function onGlassQualityChanged() { root._recapture() }
    }

    // always-present flat floor (the only thing visible when glass is off)
    Rectangle {
        anchors.fill: parent
        color: Theme.bgBase
    }

    // the static, soft source — rendered offscreen via the ShaderEffectSource only
    Item {
        id: src
        anchors.fill: parent
        visible: false
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0.0; color: Theme.bgSurface }
                GradientStop { position: 1.0; color: Theme.bgBase }
            }
        }
        // faint accent glints seen THROUGH the glass. Kept dark on purpose: on a
        // dark theme, high transparency reads as "smoked glass" over this soft
        // field — pushing these brighter lifts panel-bg luminance and breaks light
        // text contrast (verified against the label-column probe, which bounds them).
        Rectangle {
            width: parent.width * 0.85; height: width; radius: width / 2
            x: -parent.width * 0.22; y: -parent.height * 0.14
            color: Theme.accent; opacity: 0.07
        }
        Rectangle {
            width: parent.width * 0.6; height: width; radius: width / 2
            x: parent.width * 0.64; y: parent.height * 0.48
            color: Theme.accent; opacity: 0.055
        }
        Rectangle {
            width: parent.width * 0.45; height: width; radius: width / 2
            x: parent.width * 0.30; y: parent.height * 0.92
            color: Theme.accentHover; opacity: 0.035
        }
    }

    ShaderEffectSource {
        id: capture
        anchors.fill: parent
        sourceItem: src
        live: false                         // capture once; recaptured only via scheduleUpdate()
        visible: false                      // texture provider only; MultiEffect consumes it
        textureSize: Qt.size(Math.max(1, Math.round(root.width  * Theme.glassDownsample)),
                             Math.max(1, Math.round(root.height * Theme.glassDownsample)))
    }

    MultiEffect {
        anchors.fill: parent
        source: capture
        visible: Theme.glassEnabled
        blurEnabled: true
        blur: 1.0
        blurMax: Theme.glassBlurMax
        autoPaddingEnabled: false           // edges fade to the bgBase floor below — fine on dark
    }
}
