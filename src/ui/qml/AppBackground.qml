import QtQuick
import QtQuick.Effects
import App

// Flowing-light backdrop (DESIGN_BRIEF.md §4.1): four big color blobs — one per
// corner, in the four "structural" accents (mint / sky / lilac / coral) — drift in
// a slow breathing motion (InOutSine, 17–25s) under ONE gaussian blur, so the
// palette dissolves into ambience and the glass cards above read as frosted.
//
// Cost: the blob layer re-blurs as the blobs move, but it is downsampled (0.5x)
// and the motion is glacial; on any modern GPU this is negligible, and it is a
// separate layer from the real-time meters/numbers (those live SHARP in the
// panels above and are never sampled here). Glass OFF -> collapse to flat bgBase,
// animations stop (running gated on glassEnabled).
//
// Readability discipline (§4.2): blob `op` (opacity) is the contrast main-knob and
// is kept LOW — a bright backdrop would lift the (smoked-glass) panel luminance and
// eat light-text contrast. The exact values are bounded by .tmp/check_qml.py, which
// samples the composited panel background across several animation frames.
Item {
    id: root

    // always-present flat floor (the only thing visible when glass is off)
    Rectangle {
        anchors.fill: parent
        color: Theme.bgBase
    }

    // soft static base gradient under the blobs (cheap, not blurred)
    Rectangle {
        anchors.fill: parent
        visible: Theme.glassEnabled
        opacity: 0.62
        gradient: Gradient {
            GradientStop { position: 0.0; color: Theme.bgSurface }
            GradientStop { position: 1.0; color: Theme.bgBase }
        }
    }

    // the drifting color field — blurred ONCE as a single downsampled layer
    Item {
        id: field
        anchors.fill: parent
        visible: Theme.glassEnabled
        layer.enabled: Theme.glassEnabled
        layer.smooth: true
        // downsample the blur source: the blobs are huge & soft, so half-res is
        // invisible but ~4x cheaper to blur each frame.
        layer.textureSize: Qt.size(Math.max(1, Math.round(root.width  * 0.5)),
                                   Math.max(1, Math.round(root.height * 0.5)))
        layer.effect: MultiEffect {
            blurEnabled: true
            blur: 1.0
            blurMax: Theme.glassBlurMax
            autoPaddingEnabled: true
        }

        Repeater {
            model: [
                { hue: Theme.mint,  op: 0.22, x0: -0.18, y0: -0.16, sz: 0.86, ax: 0.05, ay: 0.06, px: 19000, py: 23000 },
                { hue: Theme.sky,   op: 0.19, x0:  0.60, y0: -0.10, sz: 0.70, ax: 0.06, ay: 0.05, px: 21000, py: 17000 },
                { hue: Theme.lilac, op: 0.18, x0: -0.12, y0:  0.58, sz: 0.74, ax: 0.05, ay: 0.06, px: 25000, py: 20000 },
                { hue: Theme.coral, op: 0.17, x0:  0.60, y0:  0.60, sz: 0.66, ax: 0.06, ay: 0.05, px: 18000, py: 24000 }
            ]
            delegate: Rectangle {
                id: blob
                required property var modelData
                readonly property real bx:   root.width  * modelData.x0
                readonly property real by:   root.height * modelData.y0
                readonly property real ampx: root.width  * modelData.ax
                readonly property real ampy: root.height * modelData.ay

                width: root.width * modelData.sz
                height: width
                radius: width / 2
                color: modelData.hue
                opacity: modelData.op

                SequentialAnimation on x {
                    running: Theme.glassEnabled
                    loops: Animation.Infinite
                    NumberAnimation { from: blob.bx - blob.ampx; to: blob.bx + blob.ampx
                                      duration: blob.modelData.px; easing.type: Easing.InOutSine }
                    NumberAnimation { from: blob.bx + blob.ampx; to: blob.bx - blob.ampx
                                      duration: blob.modelData.px; easing.type: Easing.InOutSine }
                }
                SequentialAnimation on y {
                    running: Theme.glassEnabled
                    loops: Animation.Infinite
                    NumberAnimation { from: blob.by - blob.ampy; to: blob.by + blob.ampy
                                      duration: blob.modelData.py; easing.type: Easing.InOutSine }
                    NumberAnimation { from: blob.by + blob.ampy; to: blob.by - blob.ampy
                                      duration: blob.modelData.py; easing.type: Easing.InOutSine }
                }
            }
        }
    }
}
