import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import App

ApplicationWindow {
    id: win
    visible: true
    width: 1280
    height: 720
    minimumWidth: 960
    minimumHeight: 600
    title: "Meloie"
    color: Theme.bgBase

    // responsive breakpoint: below this the two columns stack (brief §3 §136).
    readonly property bool narrow: width < 980

    // close (X) -> hide to the system tray (stream keeps running); the tray menu
    // 退出 truly quits. When no tray exists, trayActive is false -> normal quit.
    onClosing: function(close) {
        if (backend.trayActive) {
            close.accepted = false;
            win.hide();
        }
    }

    // -------- helpers (unchanged logic) --------
    property var micList: {
        var arr = [{ "name": "🎙 系统默认 mic", "index": -1 }];
        var devs = backend.devices;
        for (var i = 0; i < devs.length; i++)
            if (devs[i].maxIn > 0) arr.push(devs[i]);
        return arr;
    }
    property var outList: {
        var arr = [];
        var devs = backend.devices;
        for (var i = 0; i < devs.length; i++)
            if (devs[i].maxOut > 0) arr.push(devs[i]);
        return arr;
    }
    property var monitorList: {
        // headphone/speaker monitor — exclude CABLE Input so we never double-feed
        // the virtual cable (that's the main sink, picked in the 输出 combo).
        var arr = [{ "name": "🎧 系统默认输出", "index": -1 }];
        var devs = backend.devices;
        for (var i = 0; i < devs.length; i++)
            if (devs[i].maxOut > 0 && !devs[i].isCableInput) arr.push(devs[i]);
        return arr;
    }

    // saved precise-mappings for the load dropdown: a placeholder at index 0 + backend list.
    property var preciseMapList: {
        var arr = [{ "name": "（载入已保存映射…）", "file": "" }];
        var ms = backend.preciseMaps;
        for (var i = 0; i < ms.length; i++) arr.push(ms[i]);
        return arr;
    }
    function preciseSuggestName() {
        function stem(n) { return (n || "").replace(/\.wav$/i, ""); }
        var v = stem(backend.preciseVoiceName), t = stem(backend.preciseTargetName);
        return (!v && !t) ? "" : (v + "→" + t);
    }

    function modelPath() {
        var p = backend.models[modelCombo.currentIndex];
        return p ? p.path : "";
    }
    function micSubstr() {
        var d = win.micList[micCombo.currentIndex];
        return (d && d.index >= 0) ? d.name : "";
    }
    function outSubstr() {
        var d = win.outList[outCombo.currentIndex];
        return d ? d.name : "CABLE Input";
    }
    function monitorSubstr() {
        var d = win.monitorList[monitorCombo.currentIndex];
        return (d && d.index >= 0) ? d.name : "";    // "" -> system default output
    }
    function fmt(v) { return (v === undefined || v === null) ? "—" : v.toFixed(0); }

    // device-combo selection survives a list refresh (↻): a model swap resets a
    // ComboBox's currentIndex to 0, so we re-find the saved device name in the
    // rebuilt list, else fall back to a sensible default index.
    function indexByName(list, sel, fallback) {
        for (var i = 0; i < list.length; i++)
            if (list[i].name === sel) return i;
        return fallback;
    }
    function cableInputIndex(list) {
        for (var i = 0; i < list.length; i++)
            if (list[i].isCableInput) return i;
        return 0;
    }

    // ---- 融合模式 state: models to blend into the base + their weights ----
    property var mergeCandidates: {
        var base = win.modelPath();
        var arr = [];
        var ms = backend.models;
        for (var i = 0; i < ms.length; i++)
            if (ms[i].path !== base) arr.push(ms[i]);
        return arr;
    }
    property var mergePick: ({})        // path -> weight (only checked models)
    function setMergePick(path, checked, weight) {
        var m = {};
        for (var k in win.mergePick) m[k] = win.mergePick[k];
        if (checked) m[path] = weight;
        else delete m[path];
        win.mergePick = m;              // reassign so bindings (button enabled) react
    }

    // A2 auto pitch-centering availability (per-model target_f0_median seed)
    property real autoCenterTarget: 0
    property bool autoCenterAvail: false

    function initSliders() {
        var p = backend.modelParams;
        if (!p) return;
        // literal fallbacks below are last-resort only; the canonical neutral values
        // live in config_assembly.NEUTRAL_DEFAULTS (keep them in sync).
        pitchSlider.value   = p.pitch_shift    !== undefined ? p.pitch_shift    : 0;
        protectSlider.value = p.protect        !== undefined ? p.protect        : 0.33;
        indexSlider.value   = p.index_rate     !== undefined ? p.index_rate     : 0.0;
        formantSlider.value = p.formant_timbre !== undefined ? p.formant_timbre : 1.0;
        formantOn.checked   = (p.formant_on === true);   // reflect a saved gender shift
        indexSlider.enabled = (p.has_index === true);
        win.autoCenterTarget = p.target_f0_median !== undefined ? p.target_f0_median : 0;
        win.autoCenterAvail  = win.autoCenterTarget > 0;
        autoCenterOn.checked = false;        // fresh model load -> auto-center off (engine default)
    }

    // compound-control pushers (checkbox + slider(s) -> one setter call)
    function pushFormant()   { backend.setFormant(formantOn.checked, formantSlider.value); }
    function pushDenoise()   { backend.setDenoise(denoiseOn.checked, denoiseStr.value, nonstat.checked); }
    function pushSilence()   { backend.setSilenceGate(silOn.checked, silDb.value); }
    function pushAutotune()  { backend.setAutotune(atOn.checked, atStr.value); }
    function pushAutoPitch() { backend.setAutoPitch(apOn.checked, apThr.value); }

    Component.onCompleted: {
        win.initSliders();
        outCombo.currentIndex = win.cableInputIndex(win.outList);
    }

    Connections {
        target: backend
        function onModelParamsChanged() { win.initSliders(); }
        function onModelsChanged() { win.initSliders(); }
        function onModelMerged(path) {
            // a merge finished: select the new model + (if running) reload to it
            win.mergePick = ({});
            for (var i = 0; i < backend.models.length; i++)
                if (backend.models[i].path === path) { modelCombo.currentIndex = i; break; }
            backend.selectModel(win.modelPath());
            if (backend.state === "running")
                backend.reloadModel(win.modelPath(), win.micSubstr(), win.outSubstr(),
                                    f0Combo.currentText, win.monitorSubstr(), monitorSw.checked);
        }
        function onSidReset() { sidSpin.value = 0; }   // every (re)load resets engine to sid 0
        function onErrorOccurred(msg) { errorLabel.text = msg; }
        function onMetricsChanged(m) {
            monLatency.text   = fmt(m.rvc_inference_mean_ms) + " ms";   // right-column hero latency (header copy was removed as redundant)
            inMeter.dbfs  = m.input_peak_dbfs  !== undefined ? m.input_peak_dbfs  : -200;
            outMeter.dbfs = m.output_peak_dbfs !== undefined ? m.output_peak_dbfs : -200;
            inferLabel.text = "infer " + fmt(m.rvc_inference_last_ms) + "/" + fmt(m.rvc_inference_mean_ms)
                            + "/" + fmt(m.rvc_inference_max_ms) + " ms";
            underLabel.text = "underrun " + (m.startup_output_underruns || 0) + "／"
                            + (m.steady_state_output_underruns || 0);
            fbLabel.text    = "fallback " + (m.rvc_fallback_count || 0);
            queueLabel.text = "queue " + (m.max_input_queue_depth || 0);
            solaLabel.text  = "sola " + (m.rvc_sola_offset_last || 0);
            silLabel.text   = "sil-skip " + (m.rvc_silence_skipped_count || 0);
        }
    }

    // ===================== flowing-light background =====================
    AppBackground { id: appBg; anchors.fill: parent }

    // ===================== fixed glass command bar (chrome) =====================
    GlassPanel {
        id: header
        objectName: "panelHeader"
        backdrop: appBg
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: Theme.s4
        z: 2

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.s3

            Label { text: "模型"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
            AppComboBox {
                id: modelCombo
                Layout.preferredWidth: 150
                model: backend.models
                textRole: "name"
                enabled: !backend.busy      // a switch mid-load would desync knobs vs engine
                onActivated: {
                    win.mergePick = ({});       // base changed -> clear merge picks
                    backend.selectModel(win.modelPath());
                    if (backend.state === "running")
                        backend.reloadModel(win.modelPath(), win.micSubstr(), win.outSubstr(), f0Combo.currentText,
                                            win.monitorSubstr(), monitorSw.checked);
                }
            }
            AppButton { text: "↻"; flat: true; onClicked: backend.refreshModels() }   // re-scan models/ (hand-copied .pth)
            Label { text: "F0"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
            AppComboBox {
                id: f0Combo
                Layout.preferredWidth: 100
                model: ["fcpe", "rmvpe"]
                currentIndex: 0     // index 0 MUST equal config_assembly.DEFAULT_F0 ("fcpe")
                // f0 is live-swappable in loaded AND running states (no ~30s reload);
                // with nothing loaded the backend no-ops and Start picks the value up.
                onActivated: backend.setF0Method(currentText)
            }
            BusyIndicator { running: backend.busy; visible: backend.busy; implicitWidth: 22; implicitHeight: 22 }
            StatusPill { statusText: backend.state }
            Item { Layout.fillWidth: true }    // push the game-mode control to the right

            // ---- 游戏模式 (顶栏右侧三态滑块；降/清零独显占用换游戏时可用实时推理) ----
            Label { text: "游戏模式"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
            SegmentedControl {
                id: gameModeSeg
                Layout.preferredWidth: 216
                Layout.preferredHeight: 30
                enabled: !backend.busy
                dimFirst: true                 // index 0 = 关 -> neutral pill (not accent)
                property var keys: ["off", "dgpu_light", "cpu_zero"]
                options: ["关", "降dGPU", "零dGPU"]
                currentIndex: 0                // backend.gameMode starts "off" -> index 0
                onActivated: backend.setGameMode(keys[index])
                // sync if the backend changes mode itself (avoids the currentIndex
                // binding-break gotcha after a user pick).
                Connections {
                    target: backend
                    function onGameModeChanged() {
                        gameModeSeg.currentIndex =
                            Math.max(0, gameModeSeg.keys.indexOf(backend.gameMode));
                    }
                }
                HoverHandler { id: gmHover }
                ToolTip {
                    visible: gmHover.hovered
                    delay: 350
                    text: "游戏模式：降 dGPU=独显降负载(块500ms)；零 dGPU=推理移到 CPU(~8核)、独显空闲、延迟~1.3s、精度降低(自动关检索+开静音门限)。切换会重载模型(数秒)。"
                }
            }
        }
    }

    // ===================== two-column body: controls | monitor =====================
    GridLayout {
        id: body
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        // run to the window bottom so the left viewport's dissolve strip lives in the
        // Start button's bottom margin; lift only when an error needs the footer room.
        anchors.bottom: errorLabel.visible ? errorLabel.top : parent.bottom
        anchors.leftMargin: Theme.s4
        anchors.rightMargin: Theme.s4
        // top/bottom run to the header bottom / window edge so the left viewport's
        // dissolve strips can live in the monitor card's top inset / Start's bottom
        // inset. Lift the bottom only when an error needs the footer room.
        anchors.topMargin: 0
        anchors.bottomMargin: errorLabel.visible ? Theme.s2 : 0
        columns: win.narrow ? 1 : 2
        columnSpacing: Theme.s3
        rowSpacing: Theme.s3

        // ---------------- LEFT: controls (scrollable; capped so sliders stay precise) ----------------
        Item {
            id: leftCol
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.horizontalStretchFactor: 3
            Layout.maximumWidth: win.narrow ? 1000000 : 880

            ScrollView {
            id: leftScroll
            objectName: "leftScroll"
            anchors.fill: parent
            contentWidth: availableWidth
            clip: true
            ScrollBar.vertical.policy: ScrollBar.AlwaysOff   // no scrollbar rail (drift/wheel only)
            // dissolve the scroll edges INTO the backdrop instead of a hard clip
            // line: render the whole column to a layer and fade its own alpha to 0
            // over a band at each edge (via the gradient mask below) — only the edge
            // that actually has hidden content that way. The card melts into the
            // flowing-light background (no dividing line, no dark veil). Glass OFF ->
            // layer disabled -> plain hard clip. (Live-verified path: AppBackground
            // already uses layer.effect:MultiEffect on this GPU; offscreen can't
            // render the mask texture, so the headless grab will look clipped/blank
            // at the edges — that's the known offscreen limitation, not the live look.)
            layer.enabled: Theme.glassEnabled
            layer.smooth: true
            layer.effect: MultiEffect {
                autoPaddingEnabled: false
                maskEnabled: true
                maskSource: edgeFadeMask
                maskThresholdMin: 0.5
                maskSpreadAtMin: 0.5
            }

            ColumnLayout {
                width: leftScroll.availableWidth
                spacing: Theme.s3

                // top reserve mirroring the bottom: when at the very top, the first
                // card aligns with the monitor card's top edge, and the strip above it
                // (header -> first card) is the top dissolve zone. minus s3 = the
                // ColumnLayout spacing already sits below this spacer.
                Item { Layout.fillWidth: true; implicitHeight: Math.max(0, Theme.glassEdgeBand - Theme.s3) }

                // ---------------- devices ----------------
                GlassPanel {
                    objectName: "panelDevices"
                    backdrop: appBg
                    title: "设备 / Devices"
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        Label { text: "麦克风"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
                        AppComboBox {
                            id: micCombo
                            Layout.fillWidth: true
                            model: win.micList
                            textRole: "name"
                            property string selName: ""
                            onActivated: selName = win.micList[currentIndex] ? win.micList[currentIndex].name : ""
                            onModelChanged: currentIndex = win.indexByName(win.micList, selName, 0)
                        }
                        Label { text: "输出"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
                        AppComboBox {
                            id: outCombo
                            Layout.fillWidth: true
                            model: win.outList
                            textRole: "name"
                            property string selName: ""
                            onActivated: selName = win.outList[currentIndex] ? win.outList[currentIndex].name : ""
                            onModelChanged: currentIndex = win.indexByName(win.outList, selName, win.cableInputIndex(win.outList))
                        }
                        AppButton { text: "↻"; flat: true; onClicked: backend.refreshDevices() }
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fsBody
                        property string outName: win.outSubstr()
                        property bool routingOk: outName.toLowerCase().indexOf("cable input") >= 0
                        color: routingOk ? Theme.success : Theme.warning
                        text: routingOk
                              ? "✓ 路由正常：输出→CABLE Input（下游选 CABLE Output）"
                              : "⚠ 输出应为 CABLE Input 才能让下游听到"
                    }

                    // hairline divider
                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.hairline }

                    // ---- monitor (听变声后的声音；纯路由复制，契约安全) ----
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        AppSwitch {
                            id: monitorSw
                            text: "监听"
                            onToggled: backend.setMonitor(checked)
                        }
                        AppComboBox {
                            id: monitorCombo
                            Layout.fillWidth: true
                            model: win.monitorList
                            textRole: "name"
                            property string selName: ""
                            onActivated: selName = win.monitorList[currentIndex] ? win.monitorList[currentIndex].name : ""
                            onModelChanged: currentIndex = win.indexByName(win.monitorList, selName, 0)
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        color: Theme.textMuted
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fsCaption
                        text: "🎧 监听=把变声后的声音同时送到耳机（~250–400ms 延迟侧音，确认效果用）。开关实时；切换监听设备需重启 Start 生效。"
                    }
                }

                // ---------------- creative (live) ----------------
                GlassPanel {
                    objectName: "panelCreative"
                    backdrop: appBg
                    title: "创意 / Creative（实时）"
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        Label { text: "声线 sid"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody; Layout.preferredWidth: 96 }
                        AppSpinBox {
                            id: sidSpin
                            from: 0
                            to: Math.max(0, backend.numSpeakers - 1)
                            value: 0
                            editable: true
                            enabled: backend.numSpeakers > 1
                            onValueModified: backend.setSid(value)
                        }
                        Label {
                            color: Theme.textMuted
                            font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody
                            text: backend.numSpeakers > 1 ? ("共 " + backend.numSpeakers + " 个声线") : "单声线模型"
                        }
                        Item { Layout.fillWidth: true }
                    }
                    LabeledSlider {
                        id: pitchSlider; label: "变调 pitch"; from: -24; to: 24; stepSize: 1; decimals: 0; suffix: " st"
                        accentColor: Theme.pitch             // pitch dimension = sky blue
                        // auto-center AND precise mapping both REPLACE the manual transpose
                        enabled: !autoCenterOn.checked && !backend.preciseMappingOn
                        onMoved: backend.setPitch(value)
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        AppCheckBox {
                            id: autoCenterOn
                            text: "自动音高居中"
                            accentColor: Theme.pitch          // also the pitch dimension
                            enabled: win.autoCenterAvail && !backend.preciseMappingOn
                            onToggled: backend.setAutoCenter(checked)
                        }
                        Label {
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                            color: Theme.textMuted
                            font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody
                            text: win.autoCenterAvail
                                  ? ("→ " + Math.round(win.autoCenterTarget) + " Hz（替代手动变调）")
                                  : "（此模型未设目标音高）"
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        AppCheckBox { id: formantOn; text: "性别/共振"; accentColor: Theme.resonance; onToggled: win.pushFormant() }
                        LabeledSlider {
                            id: formantSlider; label: ""; from: 0.5; to: 2.0; stepSize: 0.01; decimals: 2; value: 1.0
                            accentColor: Theme.resonance      // 性别/共振 = lilac
                            enabled: formantOn.checked
                            onMoved: win.pushFormant()
                        }
                    }
                    LabeledSlider {
                        id: indexSlider; label: "检索 index"; from: 0.0; to: 1.0; stepSize: 0.01; decimals: 2
                        // a game mode owns this knob (forces index 0) -> gray out so the
                        // slider can't lie about / fight the live value while it's active.
                        enabled: backend.gameMode === "off"
                        onMoved: backend.setIndexRate(value)
                    }

                    // ---- 💾 per-model save: remember the current carrier knobs as this model's default ----
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        Label { text: "模型默认"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody; Layout.preferredWidth: 96 }
                        Item { Layout.fillWidth: true }
                        AppButton {
                            text: "💾 记住当前"; flat: true
                            onClicked: {
                                var params = {
                                    "pitch_shift": pitchSlider.value,
                                    "index_rate": indexSlider.value,
                                    "protect": protectSlider.value,
                                    "formant_timbre": formantSlider.value,
                                    "formant_on": formantOn.checked
                                };
                                if (backend.saveModelDefaults(win.modelPath(), params))
                                    errorLabel.text = "已记住 " + modelCombo.currentText + " 的设置";
                            }
                        }
                    }

                    // ---- 融合模式 / merge: blend other models into the base (offline) ----
                    Rectangle { Layout.fillWidth: true; height: 1; color: Theme.hairline }
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "🧬 融合模式"; color: Theme.resonance      // creative timbre morph = lilac
                            font.family: Theme.fontFamily; font.pixelSize: Theme.fsLabel
                            font.weight: Theme.fwSemibold; font.letterSpacing: 0.5
                        }
                        Item { Layout.fillWidth: true }
                        AppSwitch { id: mergeToggle; objectName: "mergeToggle"; checked: false }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: mergeToggle.checked
                        spacing: Theme.s2

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            Label {
                                text: "基础：" + (modelCombo.currentText || "—")
                                color: Theme.textPrimary
                                font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody
                                Layout.preferredWidth: 140
                            }
                            LabeledSlider {
                                id: baseWeight; label: "权重"; from: 0.0; to: 1.0; stepSize: 0.05; decimals: 2; value: 1.0
                                accentColor: Theme.resonance
                            }
                        }
                        Repeater {
                            model: win.mergeCandidates
                            delegate: RowLayout {
                                required property var modelData
                                Layout.fillWidth: true
                                spacing: Theme.s3
                                AppCheckBox {
                                    id: mcb
                                    text: modelData.name
                                    accentColor: Theme.resonance
                                    Layout.preferredWidth: 140
                                    onToggled: win.setMergePick(modelData.path, checked, mwSlider.value)
                                }
                                LabeledSlider {
                                    id: mwSlider; label: ""; from: 0.0; to: 1.0; stepSize: 0.05; decimals: 2; value: 1.0
                                    accentColor: Theme.resonance
                                    enabled: mcb.checked
                                    onMoved: if (mcb.checked) win.setMergePick(modelData.path, true, value)
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            Label { text: "新名字"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
                            AppTextField {
                                id: mergeName
                                Layout.fillWidth: true
                                placeholderText: "例如 A+C"
                                // mint focus scheme (Theme.accent default): border + glow ring + caret
                            }
                            AppButton {
                                text: "🧬 融合并加载"
                                accentColor: Theme.resonance
                                enabled: !backend.busy && Object.keys(win.mergePick).length > 0
                                onClicked: {
                                    errorLabel.text = "";
                                    var base = win.modelPath();
                                    var others = [];
                                    var keys = Object.keys(win.mergePick);
                                    for (var i = 0; i < keys.length; i++)
                                        if (keys[i] !== base) others.push({ "path": keys[i], "weight": win.mergePick[keys[i]] });
                                    var nm = (mergeName.text && mergeName.text.length > 0)
                                             ? mergeName.text : (modelCombo.currentText + "+merge");
                                    backend.mergeModels(base, baseWeight.value, others, nm,
                                                        pitchSlider.value, f0Combo.currentText);
                                }
                            }
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            color: Theme.textMuted
                            font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption
                            text: "只能融合同采样率/架构的模型；融合需几秒，完成后新音色出现在顶部下拉并自动选中。"
                        }
                    }
                }

                // ---------------- advanced (collapsible, live) ----------------
                GlassPanel {
                    objectName: "panelAdvanced"
                    backdrop: appBg
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "高级 / Advanced"; color: Theme.textSecond
                            font.family: Theme.fontFamily; font.pixelSize: Theme.fsLabel; font.weight: Theme.fwSemibold; font.letterSpacing: 0.5
                        }
                        Item { Layout.fillWidth: true }
                        AppSwitch { id: advToggle; objectName: "advToggle"; checked: false }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: advToggle.checked
                        spacing: Theme.s2

                        LabeledSlider {
                            id: protectSlider; label: "protect"; from: 0.0; to: 0.5; stepSize: 0.01; decimals: 2
                            onMoved: backend.setProtect(value)
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            AppCheckBox { id: denoiseOn; text: "输入降噪"; onToggled: win.pushDenoise() }
                            LabeledSlider {
                                id: denoiseStr; label: "strength"; from: 0.0; to: 1.0; stepSize: 0.05; decimals: 2; value: 0.5
                                enabled: denoiseOn.checked; onMoved: win.pushDenoise()
                            }
                            AppCheckBox { id: nonstat; text: "nonstat"; checked: true; enabled: denoiseOn.checked; onToggled: win.pushDenoise() }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            // a game mode owns the silence gate (forces it on) -> gray out while active.
                            AppCheckBox { id: silOn; text: "静音门限"; enabled: backend.gameMode === "off"; onToggled: win.pushSilence() }
                            LabeledSlider {
                                id: silDb; label: "dBFS"; from: -80; to: -20; stepSize: 1; decimals: 0; value: -50
                                enabled: silOn.checked && backend.gameMode === "off"; onMoved: win.pushSilence()
                            }
                            // honest display while a game mode forces the gate on at -45:
                            // show that, and restore the USER'S UI state on exit (the
                            // backend restores the engine itself; this mirrors the UI).
                            // prevMode guards active<->active switches from re-saving.
                            Connections {
                                target: backend
                                property string prevMode: "off"
                                property bool userOn: false
                                property real userDb: -50
                                function onGameModeChanged() {
                                    var m = backend.gameMode;
                                    if (m !== "off" && prevMode === "off") {
                                        userOn = silOn.checked;
                                        userDb = silDb.value;
                                        silOn.checked = true;
                                        silDb.value = -45;
                                    } else if (m === "off" && prevMode !== "off") {
                                        silOn.checked = userOn;
                                        silDb.value = userDb;
                                    }
                                    prevMode = m;
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            AppCheckBox { id: atOn; text: "autotune"; enabled: !backend.preciseMappingOn; onToggled: win.pushAutotune() }
                            LabeledSlider {
                                id: atStr; label: "strength"; from: 0.0; to: 1.0; stepSize: 0.05; decimals: 2; value: 1.0
                                enabled: atOn.checked && !backend.preciseMappingOn; onMoved: win.pushAutotune()
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            AppCheckBox { id: apOn; text: "auto-pitch"; enabled: !backend.preciseMappingOn; onToggled: win.pushAutoPitch() }
                            LabeledSlider {
                                id: apThr; label: "Hz"; from: 80; to: 300; stepSize: 1; decimals: 0; value: 155
                                enabled: apOn.checked && !backend.preciseMappingOn; onMoved: win.pushAutoPitch()
                            }
                        }

                        // hairline divider
                        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.hairline }

                        // ---- appearance: glass enhancement (pure UI, no backend) ----
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: Theme.s3
                            AppSwitch {
                                id: glassSw
                                text: "玻璃质感"
                                checked: Theme.glassEnabled
                                onToggled: Theme.glassEnabled = checked
                            }
                            Item { Layout.fillWidth: true }
                            Label { text: "质量"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
                            AppComboBox {
                                Layout.preferredWidth: 110
                                model: ["high", "medium", "low"]
                                enabled: Theme.glassEnabled
                                currentIndex: 0
                                onActivated: Theme.glassQuality = currentText
                            }
                        }
                    }
                }
                // bottom reserve so a fully-scrolled last card rests flush on the
                // Start-button line: net gap below it must == glassEdgeBand (== Start's
                // bottom inset). Subtract s3 — the ColumnLayout spacing already sits
                // above this spacer and counts toward that gap.
                Item { Layout.fillWidth: true; implicitHeight: Math.max(0, Theme.glassEdgeBand - Theme.s3) }
            }
            }

            // alpha mask driving the dissolve above: white (= keep) through the
            // middle, ramping to transparent (= melt away) over a `band`-tall strip
            // at each edge — but ONLY at an edge that currently has hidden content
            // (a card at rest never fades). visible:false + layer.enabled -> this is
            // consumed purely as a texture by leftScroll's MultiEffect.maskSource.
            Rectangle {
                id: edgeFadeMask
                anchors.fill: parent
                visible: false
                layer.enabled: true
                layer.smooth: true
                readonly property real bandFrac:
                    Math.min(0.45, Theme.glassEdgeBand / Math.max(1, leftScroll.height))
                gradient: Gradient {
                    GradientStop {
                        position: 0.0
                        color: leftScroll.contentItem.atYBeginning ? "white" : "transparent"
                        Behavior on color { ColorAnimation { duration: Theme.durBase } }
                    }
                    GradientStop { position: edgeFadeMask.bandFrac; color: "white" }
                    GradientStop { position: 1.0 - edgeFadeMask.bandFrac; color: "white" }
                    GradientStop {
                        position: 1.0
                        color: leftScroll.contentItem.atYEnd ? "white" : "transparent"
                        Behavior on color { ColorAnimation { duration: Theme.durBase } }
                    }
                }
            }
        }

        // ---------------- RIGHT: live monitor (always visible) + Start/Stop CTA at the bottom ----------------
        ColumnLayout {
            id: rightCol
            Layout.fillWidth: true
            Layout.fillHeight: !win.narrow      // wide: fill so the CTA pins to the bottom; narrow: compact
            Layout.alignment: Qt.AlignTop
            Layout.horizontalStretchFactor: 2
            Layout.minimumWidth: win.narrow ? 0 : 280
            // inset the monitor card DOWN (top) and the Start button UP (bottom) by the
            // dissolve band, so the strips above the monitor top and below the Start
            // bottom are where the left column melts. The left content carries matching
            // top/bottom reserves -> its first/last card align with the monitor-top /
            // Start-bottom lines. Same token as leftScroll's mask -> always aligned.
            Layout.topMargin: win.narrow ? 0 : Theme.glassEdgeBand
            Layout.bottomMargin: win.narrow ? 0 : Theme.glassEdgeBand
            spacing: Theme.s3

            GlassPanel {
                id: monitorPanel
                objectName: "panelMonitor"
                backdrop: appBg
                thin: true
                title: "实时监控 / Monitor"
                Layout.fillWidth: true

                // hero metric: live inference latency (the monitor's headline reading)
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s2
                    Label {
                        id: monLatency; text: "— ms"
                        color: Theme.accent; font.family: Theme.fontFamily
                        font.pixelSize: 30; font.weight: Theme.fwSemibold
                    }
                    Label {
                        text: "推理延迟 / latency"; color: Theme.textMuted
                        font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption
                        Layout.alignment: Qt.AlignBottom; Layout.bottomMargin: Theme.s1
                    }
                    Item { Layout.fillWidth: true }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.hairline; Layout.bottomMargin: Theme.s1 }

                LevelMeter { id: inMeter;  label: "IN";  dbfs: -200; baseColor: Theme.input }   // 输入能量 = 珊瑚
                LevelMeter { id: outMeter; label: "OUT"; dbfs: -200; baseColor: Theme.accent }  // 输出/正向 = 青

                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.hairline; Layout.topMargin: Theme.s1 }

                Flow {
                    Layout.fillWidth: true
                    spacing: Theme.s4
                    Label { id: inferLabel; text: "infer —/—/— ms"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption }
                    Label { id: underLabel; text: "underrun 0／0";    color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption }
                    Label { id: fbLabel;    text: "fallback 0";       color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption }
                    Label { id: queueLabel; text: "queue 0";          color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption }
                    Label { id: solaLabel;  text: "sola 0";           color: Theme.textMuted;  font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption }
                    Label { id: silLabel;   text: "sil-skip 0";       color: Theme.textMuted;  font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption }
                }
            }

            // ---------------- 精确映射 / Precise Mapping（输入端 F0 分布匹配，CDF）----------------
            GlassPanel {
                id: precisePanel
                objectName: "panelPrecise"
                backdrop: appBg
                title: "精确映射 / Precise Mapping"
                Layout.fillWidth: true
                // content-sized + the fillHeight spacer below pins Start to the bottom,
                // so this card sits in the UPPER portion of the free area (well under ½).

                // ready to 启用 = a map already in hand (loaded/built; needs no engine), OR
                // two wavs picked + a loaded model (a fresh build needs the f0 estimator).
                readonly property bool ready:
                    !backend.busy && (backend.precisePending
                        || (backend.preciseVoiceName.length > 0 && backend.preciseTargetName.length > 0
                            && (backend.state === "loaded" || backend.state === "running")))

                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s3
                    AppSwitch {
                        id: preciseSw
                        text: "启用"
                        checked: backend.preciseMappingOn
                        enabled: precisePanel.ready || backend.preciseMappingOn
                        onToggled: backend.setPreciseMapping(checked)
                        // a user toggle breaks the checked binding (same gotcha as
                        // gameModeSeg) -> re-sync when the backend flips the mapping
                        // itself (failed build, wav re-pick, model-swap reset).
                        Connections {
                            target: backend
                            function onPreciseChanged() { preciseSw.checked = backend.preciseMappingOn; }
                        }
                    }
                    Item { Layout.fillWidth: true }
                }
                // load a saved mapping (skip re-picking wavs + rebuilding). Selecting only
                // STAGES it -> flip 启用 to apply.
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s3
                    Label { text: "已保存"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
                    AppComboBox {
                        id: preciseMapCombo
                        Layout.fillWidth: true
                        model: win.preciseMapList
                        textRole: "name"
                        onActivated: {
                            if (currentIndex > 0)
                                backend.loadPreciseMap(win.preciseMapList[currentIndex].file);
                        }
                    }
                    AppButton { text: "↻"; flat: true; onClicked: backend.refreshPreciseMaps() }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s3
                    AppButton {
                        text: "🎙 你的声音"; flat: true
                        onClicked: backend.choosePreciseVoiceWav()
                    }
                    Label {
                        Layout.fillWidth: true; elide: Text.ElideMiddle
                        color: backend.preciseVoiceName.length > 0 ? Theme.textPrimary : Theme.textMuted
                        font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody
                        text: backend.preciseVoiceName.length > 0 ? backend.preciseVoiceName : "未选择 .wav"
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s3
                    AppButton {
                        text: "🎯 模型原声"; flat: true
                        onClicked: backend.choosePreciseTargetWav()
                    }
                    Label {
                        Layout.fillWidth: true; elide: Text.ElideMiddle
                        color: backend.preciseTargetName.length > 0 ? Theme.textPrimary : Theme.textMuted
                        font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody
                        text: backend.preciseTargetName.length > 0 ? backend.preciseTargetName : "未选择 .wav"
                    }
                }
                // save the current map (loaded or built) under an editable name -> dropdown
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s3
                    AppTextField {
                        id: preciseNameField
                        Layout.fillWidth: true
                        text: win.preciseSuggestName()       // prefilled; user can override
                        placeholderText: "映射名字"
                    }
                    AppButton {
                        text: "💾 保存"; flat: true
                        enabled: backend.precisePending && !backend.busy
                        onClicked: {
                            if (backend.savePreciseMap(preciseNameField.text))
                                errorLabel.text = "已保存映射：" + preciseNameField.text;
                        }
                    }
                }
                Label {                       // 状态（构建结果 / 进行中）
                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                    visible: backend.preciseStatus.length > 0
                    color: Theme.accent
                    font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption
                    text: backend.preciseStatus
                }
                Label {                       // 小字提示：要提供什么，否则无正面效果
                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                    color: Theme.textMuted
                    font.family: Theme.fontFamily; font.pixelSize: Theme.fsCaption
                    text: "用 CDF 把你的音高分布精确对齐到模型音域。需：①你本人一段语音（与实际使用同一人、≥2 秒有声）②模型目标声音干净样本（≥2 秒有声），并先 Start 加载模型。启用后会接管并置灰手动变调/自动音高居中/autotune/auto-pitch；样本太短/有声不足会失败，起不到正面效果。"
                }
            }

            // spacer pushes the transport CTA to the bottom of the monitor column
            Item { Layout.fillWidth: true; Layout.fillHeight: true }

            // primary transport: Start/Stop, full column width (same as the monitor card above)
            AppButton {
                id: startBtn
                text: backend.state === "running" ? "⏹ Stop" : "▶ Start"
                // semantic: Start = mint (forward/positive), Stop = coral (stop/input energy)
                accentColor: backend.state === "running" ? Theme.input : Theme.accent
                enabled: !backend.busy && modelCombo.count > 0
                Layout.fillWidth: true
                Layout.preferredHeight: 46
                onClicked: {
                    errorLabel.text = "";
                    backend.startOrStop(win.modelPath(), win.micSubstr(), win.outSubstr(), f0Combo.currentText,
                                        win.monitorSubstr(), monitorSw.checked);
                }
            }
        }
    }

    // ===================== footer: error / status message (full width) =====================
    Label {
        id: errorLabel
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.leftMargin: Theme.s4
        anchors.rightMargin: Theme.s4
        anchors.bottomMargin: Theme.s3
        height: text.length > 0 ? implicitHeight : 0
        visible: text.length > 0
        wrapMode: Text.WordWrap
        color: Theme.error
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fsBody
        text: ""
    }
}
