import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import App

ApplicationWindow {
    id: win
    visible: true
    width: 1280
    height: 720
    minimumWidth: 960
    minimumHeight: 600
    title: "RVC Voice Changer"
    color: Theme.bgBase

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
            latencyLabel.text = fmt(m.rvc_inference_mean_ms) + " ms";
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

    // ===================== background (static, blurred once) =====================
    AppBackground { anchors.fill: parent }

    // ===================== fixed glass header (chrome) =====================
    GlassPanel {
        id: header
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
                onActivated: {
                    win.mergePick = ({});       // base changed -> clear merge picks
                    backend.selectModel(win.modelPath());
                    if (backend.state === "running")
                        backend.reloadModel(win.modelPath(), win.micSubstr(), win.outSubstr(), f0Combo.currentText,
                                            win.monitorSubstr(), monitorSw.checked);
                }
            }
            Label { text: "F0"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
            AppComboBox {
                id: f0Combo
                Layout.preferredWidth: 100
                model: ["fcpe", "rmvpe"]
                currentIndex: 0     // index 0 MUST equal config_assembly.DEFAULT_F0 ("fcpe")
                onActivated: {
                    if (backend.state === "running")
                        backend.reloadModel(win.modelPath(), win.micSubstr(), win.outSubstr(), f0Combo.currentText,
                                            win.monitorSubstr(), monitorSw.checked);
                }
            }
            AppButton {
                id: startBtn
                text: backend.state === "running" ? "⏹ Stop" : "▶ Start"
                enabled: !backend.busy && modelCombo.count > 0
                onClicked: {
                    errorLabel.text = "";
                    backend.startOrStop(win.modelPath(), win.micSubstr(), win.outSubstr(), f0Combo.currentText,
                                        win.monitorSubstr(), monitorSw.checked);
                }
            }
            BusyIndicator { running: backend.busy; visible: backend.busy; implicitWidth: 22; implicitHeight: 22 }
            StatusPill { statusText: backend.state }
            Item { Layout.fillWidth: true }
            Label {
                id: latencyLabel; text: "— ms"
                color: Theme.accent; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody; font.weight: Theme.fwMedium
            }
        }
    }

    // ===================== scrolling content =====================
    ScrollView {
        id: scroll
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.leftMargin: Theme.s4
        anchors.rightMargin: Theme.s4
        anchors.topMargin: Theme.s3
        anchors.bottomMargin: Theme.s4
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: scroll.availableWidth
            spacing: Theme.s3

            // ---------------- devices ----------------
            GlassPanel {
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
                    enabled: !autoCenterOn.checked       // auto-center REPLACES manual transpose
                    onMoved: backend.setPitch(value)
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.s3
                    AppCheckBox {
                        id: autoCenterOn
                        text: "自动音高居中"
                        enabled: win.autoCenterAvail
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
                    AppCheckBox { id: formantOn; text: "性别/共振"; onToggled: win.pushFormant() }
                    LabeledSlider {
                        id: formantSlider; label: ""; from: 0.5; to: 2.0; stepSize: 0.01; decimals: 2; value: 1.0
                        enabled: formantOn.checked
                        onMoved: win.pushFormant()
                    }
                }
                LabeledSlider {
                    id: indexSlider; label: "检索 index"; from: 0.0; to: 1.0; stepSize: 0.01; decimals: 2
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
                        text: "🧬 融合模式"; color: Theme.textSecond
                        font.family: Theme.fontFamily; font.pixelSize: Theme.fsLabel
                        font.weight: Theme.fwSemibold; font.letterSpacing: 0.5
                    }
                    Item { Layout.fillWidth: true }
                    AppSwitch { id: mergeToggle; checked: false }
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
                                Layout.preferredWidth: 140
                                onToggled: win.setMergePick(modelData.path, checked, mwSlider.value)
                            }
                            LabeledSlider {
                                id: mwSlider; label: ""; from: 0.0; to: 1.0; stepSize: 0.05; decimals: 2; value: 1.0
                                enabled: mcb.checked
                                onMoved: if (mcb.checked) win.setMergePick(modelData.path, true, value)
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        Label { text: "新名字"; color: Theme.textSecond; font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody }
                        TextField {
                            id: mergeName
                            Layout.fillWidth: true
                            placeholderText: "例如 A+C"
                            color: Theme.textPrimary
                            font.family: Theme.fontFamily; font.pixelSize: Theme.fsBody
                        }
                        AppButton {
                            text: "🧬 融合并加载"
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
                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "高级 / Advanced"; color: Theme.textSecond
                        font.family: Theme.fontFamily; font.pixelSize: Theme.fsLabel; font.weight: Theme.fwSemibold; font.letterSpacing: 0.5
                    }
                    Item { Layout.fillWidth: true }
                    AppSwitch { id: advToggle; checked: false }
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
                        AppCheckBox { id: silOn; text: "静音门限"; onToggled: win.pushSilence() }
                        LabeledSlider {
                            id: silDb; label: "dBFS"; from: -80; to: -20; stepSize: 1; decimals: 0; value: -50
                            enabled: silOn.checked; onMoved: win.pushSilence()
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        AppCheckBox { id: atOn; text: "autotune"; onToggled: win.pushAutotune() }
                        LabeledSlider {
                            id: atStr; label: "strength"; from: 0.0; to: 1.0; stepSize: 0.05; decimals: 2; value: 1.0
                            enabled: atOn.checked; onMoved: win.pushAutotune()
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.s3
                        AppCheckBox { id: apOn; text: "auto-pitch"; onToggled: win.pushAutoPitch() }
                        LabeledSlider {
                            id: apThr; label: "Hz"; from: 80; to: 300; stepSize: 1; decimals: 0; value: 155
                            enabled: apOn.checked; onMoved: win.pushAutoPitch()
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

            // ---------------- telemetry (read-only) ----------------
            GlassPanel {
                title: "遥测 / Telemetry"
                LevelMeter { id: inMeter;  label: "IN";  dbfs: -200 }
                LevelMeter { id: outMeter; label: "OUT"; dbfs: -200 }
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

            Label {
                id: errorLabel
                Layout.fillWidth: true
                Layout.leftMargin: 4
                wrapMode: Text.WordWrap
                color: Theme.error
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fsBody
                text: ""
            }
        }
    }
}
