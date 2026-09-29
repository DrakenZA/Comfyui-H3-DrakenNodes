import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Draken.PersonSelection",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DrakenPersonSelection") return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            const node = this;
            const selection = node.widgets.find(w => w.name === "selection");
            const frameIndex = node.widgets.find(w => w.name === "frame_index");
            const container = document.createElement("div");
            container.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;background:#242424;color:#eee;width:100%;";
            const toolbar = document.createElement("div");
            toolbar.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
            const mode = document.createElement("select");
            for (const [value, label] of [["box", "Drag box"], ["include", "Include point"], ["exclude", "Exclude point"]]) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = label;
                mode.append(option);
            }
            const clear = document.createElement("button");
            clear.textContent = "Clear selection";
            toolbar.append(mode, clear);
            const hint = document.createElement("div");
            hint.style.cssText = "font:12px sans-serif;line-height:1.3;";
            hint.textContent = "Queue once to load the selected frame.";
            const canvas = document.createElement("canvas");
            canvas.dataset.drakenPersonSelector = "true";
            canvas.setAttribute("aria-label", "Person selection preview");
            canvas.width = 640;
            canvas.height = 360;
            canvas.style.cssText = "width:100%;height:auto;flex-shrink:0;cursor:crosshair;touch-action:none;display:none;";
            container.append(toolbar, hint, canvas);
            let image = null;
            let displayedFrame = null;
            let requestId = 0;
            let dragging = null;
            const ctx = canvas.getContext("2d");
            const read = () => {
                try { return JSON.parse(selection.value); }
                catch { return { box: null, points: [] }; }
            };
            const draw = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (!image) return;
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                const data = read();
                const box = dragging ? [Math.min(dragging.start[0], dragging.end[0]), Math.min(dragging.start[1], dragging.end[1]),
                    Math.max(dragging.start[0], dragging.end[0]), Math.max(dragging.start[1], dragging.end[1])] : data.box;
                if (Array.isArray(box) && box.length === 4) {
                    ctx.strokeStyle = "#ffcc00";
                    ctx.lineWidth = Math.max(2, canvas.width / 220);
                    ctx.strokeRect(box[0] * canvas.width, box[1] * canvas.height,
                        (box[2] - box[0]) * canvas.width, (box[3] - box[1]) * canvas.height);
                }
                for (const point of data.points || []) {
                    ctx.beginPath();
                    ctx.arc(point[0] * canvas.width, point[1] * canvas.height, Math.max(5, canvas.width / 100), 0, 2 * Math.PI);
                    ctx.fillStyle = point[2] === 1 ? "#00ff66" : "#ff3344";
                    ctx.fill();
                    ctx.strokeStyle = "#111";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            };
            const write = data => {
                selection.value = JSON.stringify(data);
                selection.callback?.(selection.value);
                app.graph?.change();
                node.setDirtyCanvas(true, true);
                draw();
            };
            const position = event => {
                const rect = canvas.getBoundingClientRect();
                return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
                    Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
            };
            clear.onclick = () => write({ box: null, points: [] });
            canvas.onpointerdown = event => {
                if (event.button !== 0 || !image) return;
                if (displayedFrame !== Number(frameIndex.value)) {
                    hint.textContent = "Queue to refresh this frame before selecting.";
                    return;
                }
                event.preventDefault();
                const point = position(event);
                if (mode.value === "box") {
                    dragging = { start: point, end: point };
                    canvas.setPointerCapture(event.pointerId);
                    draw();
                } else {
                    const data = read();
                    write({ box: data.box ?? null, points: [...(data.points || []), [...point, mode.value === "include" ? 1 : 0]] });
                }
            };
            canvas.onpointermove = event => {
                if (!dragging) return;
                dragging.end = position(event);
                draw();
            };
            canvas.onpointerup = event => {
                if (!dragging) return;
                const end = position(event);
                const start = dragging.start;
                dragging = null;
                const box = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])];
                if (box[2] - box[0] > 0.002 && box[3] - box[1] > 0.002) write({ box, points: read().points || [] });
                else draw();
            };
            canvas.onpointercancel = () => { dragging = null; draw(); };
            const originalSelectionCallback = selection.callback;
            selection.callback = function () { originalSelectionCallback?.apply(this, arguments); draw(); };
            const originalFrameCallback = frameIndex.callback;
            frameIndex.callback = function () {
                originalFrameCallback?.apply(this, arguments);
                if (displayedFrame !== null && displayedFrame !== Number(frameIndex.value)) {
                    write({ box: null, points: [] });
                    hint.textContent = "Frame changed. Queue to refresh, then select the person again.";
                }
            };
            const previewHeight = () => image ? Math.round((node.size[0] - 32) * canvas.height / canvas.width) + 74 : 70;
            const widget = node.addDOMWidget("person_preview", "draken_person_preview", container, {
                serialize: false, getHeight: previewHeight, getMinHeight: previewHeight, getMaxHeight: previewHeight,
            });
            widget.computeSize = width => [width, previewHeight()];
            const originalExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                originalExecuted?.apply(this, arguments);
                const preview = message.person_selector?.[0];
                if (!preview) return;
                const currentRequest = ++requestId;
                const loaded = new Image();
                loaded.onload = () => {
                    if (currentRequest !== requestId) return;
                    image = loaded;
                    displayedFrame = preview.frame_index;
                    canvas.width = loaded.naturalWidth;
                    canvas.height = loaded.naturalHeight;
                    canvas.style.display = "block";
                    hint.textContent = `Frame ${preview.frame_index} of ${preview.frame_count - 1}. Select the person, then queue again.`;
                    node.setSize([node.size[0], node.computeSize()[1]]);
                    node.setDirtyCanvas(true, true);
                    draw();
                };
                loaded.onerror = () => { hint.textContent = "Preview could not load. Queue again to refresh."; };
                loaded.src = api.apiURL("/view?" + new URLSearchParams({ filename: preview.filename,
                    subfolder: preview.subfolder, type: preview.type }));
            };
            node.setSize([Math.max(node.size[0], 440), node.size[1]]);
            node.setSize([node.size[0], node.computeSize()[1]]);
        };
    },
});
