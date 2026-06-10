import QtQuick
import QtQuick.Effects
import App

// Progressive gaussian blur at a scroll boundary (the DESIGN_BRIEF "natural
// transition", per user request): instead of a hard clip line where the cards get
// razor-cut, a LIVE snapshot of the scroll's edge strip is gaussian-blurred and
// faded INWARD by a gradient mask, then layered over the sharp content — so the
// last band of content "melts" out of focus the way it dissolves under an iOS nav
// bar. Technique = ShaderEffectSource -> MultiEffect(blur + mask) per GLASS_BLUR_GUIDE.
//
// No solid backing: an earlier bgBase->transparent fallback Rectangle was a *dark
// veil* (Qt's "transparent" is transparent BLACK, so the whole band read as a near-
// black slab over the lighter backdrop). The blurred strip alone is the soft melt;
// if a GPU path can't produce the SES->MultiEffect texture the edge simply falls
// back to a clean hard clip (never a dark band). Collapses to nothing when glass OFF.
//
// Cost: one live ShaderEffectSource + one MultiEffect blur per ACTIVE edge, re-grabbed
// each frame only while there is hidden content that direction (live gated on
// hasHidden). The left scroll holds no real-time meters (those live in the right
// panel), so an idle strip is a static re-grab. Per project direction, perf is not
// the binding constraint here.
Item {
    id: edge

    // ---- wiring (set by the parent) ----
    property Item scrollView                  // the ScrollView to snapshot
    property bool topEdge: true               // true = top boundary, false = bottom
    property int  band: Theme.glassEdgeBand   // height of the soft band
    property int  blurMax: Theme.glassEdgeBlurMax

    // show only when content is actually scrolled off this direction (so the first
    // card's title isn't permanently smudged, and the bottom isn't blurred at rest)
    readonly property bool hasHidden:
        (scrollView && scrollView.contentItem)
            ? (topEdge ? !scrollView.contentItem.atYBeginning
                       : !scrollView.contentItem.atYEnd)
            : false

    anchors.left: parent.left
    anchors.right: parent.right
    height: band

    visible: Theme.glassEnabled
    opacity: hasHidden ? 1.0 : 0.0
    Behavior on opacity { NumberAnimation { duration: Theme.durFast } }

    // (1) the gaussian-blurred snapshot of the edge strip, faded inward by a mask
    Item {
        id: blurred
        anchors.fill: parent
        layer.enabled: true
        layer.smooth: true
        layer.effect: MultiEffect {
            autoPaddingEnabled: false        // edge strip: must be false or the blur bleeds out
            blurEnabled: true
            blur: 1.0
            blurMax: edge.blurMax
            maskEnabled: true
            maskSource: edgeMask
            maskThresholdMin: 0.5            // full-range smoothstep: sharp inward, blurred at edge
            maskSpreadAtMin: 0.5
        }

        ShaderEffectSource {
            anchors.fill: parent
            sourceItem: edge.scrollView
            live: edge.hasHidden && Theme.glassEnabled   // stop re-grabbing when idle/hidden
            recursive: false
            hideSource: false
            sourceRect: edge.scrollView
                ? (edge.topEdge
                   ? Qt.rect(0, 0, edge.scrollView.width, edge.band)
                   : Qt.rect(0, edge.scrollView.height - edge.band,
                             edge.scrollView.width, edge.band))
                : Qt.rect(0, 0, 1, 1)
        }
    }

    // (2) mask: alpha 1 at the very edge -> 0 inward, so blur is full at the boundary
    //     and dissolves into the sharp content (alpha read by MultiEffect.maskSource).
    Rectangle {
        id: edgeMask
        anchors.fill: parent
        visible: false
        layer.enabled: true
        layer.smooth: true
        gradient: Gradient {
            GradientStop { position: edge.topEdge ? 0.0 : 1.0; color: "white" }
            GradientStop { position: edge.topEdge ? 1.0 : 0.0; color: "transparent" }
        }
    }
}
