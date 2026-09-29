/* Optional live ComfyUI/Playwright check. A local Chrome and playwright are required.
 * Copy EdgeTAM's bedroom/00003.jpg to ComfyUI/input/person-test.jpg first.
 * node tests/person_ui_smoke.cjs http://127.0.0.1:8188 /path/to/test-results
 * Uses a new browser context, queues only its own graph, and writes its own PNG outputs.
 */
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const url = process.argv[2] || "http://127.0.0.1:8188";
const output = process.argv[3] || ".";

async function queue(page) {
    const responsePromise = page.waitForResponse(r => r.url().endsWith("/prompt") && r.request().method() === "POST");
    await page.evaluate(() => window.comfyAPI.app.app.queuePrompt(0, 1));
    const response = await responsePromise;
    const body = await response.json();
    assert.ok(body.prompt_id, JSON.stringify(body));
    let history;
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
        const data = await (await page.request.get(url + "/history/" + body.prompt_id)).json();
        history = data[body.prompt_id];
        if (history?.status?.completed || history?.status?.status_str === "error") break;
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    assert.ok(history?.status?.completed, "Prompt did not finish within 120 seconds");
    assert.equal(history.status.status_str, "success", JSON.stringify(history.status));
    return history;
}

(async () => {
    fs.mkdirSync(output, { recursive: true });
    const browser = await chromium.launch({ channel: "chrome", headless: true });
    try {
        const page = await browser.newPage({ viewport: { width: 1700, height: 1000 } });
        const errors = [];
        page.on("pageerror", error => errors.push(error.message));
        await page.goto(url, { waitUntil: "domcontentloaded" });
        await page.waitForFunction(() => !!window.comfyAPI?.app?.app?.graph &&
            !!window.LiteGraph?.registered_node_types?.DrakenPersonSelection, {}, { timeout: 45000 });
        // The frontend finishes opening its initial workspace after node registration.
        await page.waitForTimeout(3000);
        const ids = await page.evaluate(() => {
            const app = window.comfyAPI.app.app;
            app.graph.clear();
            const types = ["LoadImage", "DrakenPersonSelection", "DrakenPersonVideoMask", "DrakenPersonMaskOverlay", "SaveImage"];
            const nodes = types.map((type, i) => {
                const node = LiteGraph.createNode(type);
                app.graph.add(node); node.pos = [40 + i * 450, 140]; return node;
            });
            const [load, selection, mask, overlay, save] = nodes;
            load.widgets.find(w => w.name === "image").value = "person-test.jpg";
            mask.widgets.find(w => w.name === "device").value = "cpu";
            save.widgets.find(w => w.name === "filename_prefix").value = "person_ui_test/frames";
            load.connect(0, selection, 0); load.connect(0, mask, 0); load.connect(0, overlay, 0);
            selection.connect(0, mask, 1); mask.connect(0, overlay, 1); overlay.connect(1, save, 0);
            app.canvas.ds.offset = [0, 0]; app.canvas.ds.scale = 0.75;
            app.canvas.draw(true, true);
            window.personTestIds = Object.fromEntries(nodes.map((n, i) => [types[i], n.id]));
            return window.personTestIds;
        });
        const first = await queue(page);
        assert.ok(first.outputs[ids.DrakenPersonSelection]?.person_selector);
        assert.ok(!first.outputs[ids.SaveImage], "Empty selection must block the tracker/overlay/save without a graph error");
        const canvas = page.locator("canvas[data-draken-person-selector]");
        await canvas.waitFor({ state: "visible" });
        await page.waitForFunction(() => [...document.querySelectorAll("div")].some(e => e.textContent === "Frame 0 of 0. Select the person, then queue again."));
        const bounds = await canvas.boundingBox();
        await page.mouse.move(bounds.x + bounds.width * 0.3125, bounds.y + bounds.height * 0.074);
        await page.mouse.down();
        await page.mouse.move(bounds.x + bounds.width * 0.54, bounds.y + bounds.height * 0.9, { steps: 8 });
        await page.mouse.up();
        const container = page.getByRole("button", { name: "Clear selection" }).locator("..").locator("..");
        await container.locator("select").selectOption("include");
        await canvas.click({ position: { x: bounds.width * 0.42, y: bounds.height * 0.4 } });
        await container.locator("select").selectOption("exclude");
        await canvas.click({ position: { x: bounds.width * 0.23, y: bounds.height * 0.5 } });
        const selected = await page.evaluate(() => JSON.parse(window.comfyAPI.app.app.graph.getNodeById(
            window.personTestIds.DrakenPersonSelection).widgets.find(w => w.name === "selection").value));
        assert.ok(Math.abs(selected.box[0] - 0.3125) < 0.005, JSON.stringify(selected));
        assert.deepEqual(selected.points.map(p => p[2]), [1, 0]);
        const color = await queue(page);
        assert.ok(color.outputs[ids.SaveImage]?.images?.length === 1);
        await page.screenshot({ path: path.join(output, "comfy-person-selector.png") });
        await page.evaluate(() => {
            const node = window.comfyAPI.app.app.graph.getNodeById(window.personTestIds.DrakenPersonMaskOverlay);
            node.widgets.find(w => w.name === "mode").value = "transparent";
        });
        const transparent = await queue(page);
        assert.ok(transparent.outputs[ids.SaveImage]?.images?.length === 1);
        const cached = transparent.status.messages.find(([name]) => name === "execution_cached")?.[1]?.nodes || [];
        assert.ok(cached.includes(String(ids.DrakenPersonVideoMask)), "Changing overlay mode must reuse the tracked mask");
        fs.writeFileSync(path.join(output, "comfy-person-ui-test.json"), JSON.stringify({ selected, ids,
            color_image: color.outputs[ids.SaveImage].images[0], transparent_image: transparent.outputs[ids.SaveImage].images[0],
            cached_nodes_after_mode_change: cached, browser_errors: errors }, null, 2));
        await page.getByRole("button", { name: "Clear selection" }).click();
        const cleared = await page.evaluate(() => JSON.parse(window.comfyAPI.app.app.graph.getNodeById(
            window.personTestIds.DrakenPersonSelection).widgets.find(w => w.name === "selection").value));
        assert.deepEqual(cleared, { box: null, points: [] });
        assert.deepEqual(errors, []);
        console.log("PASS: first-queue preview, pointer box/points, real inference, saved frames, mask cache reuse, clear control");
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exit(1); });
