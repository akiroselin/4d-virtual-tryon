#!/usr/bin/env node
// Render frames of a web/scenes/<scene> outfit with the Three.js WebGPU viewer
// in headless Chromium and save them as PNG files.
//
//   node scripts/capture_webgpu.mjs --scene <name> --out <dir> [--phases 0.5] [--yaws 0,35,180]
//
// Frames are read back from the WebGPU canvas inside the page
// (window.__fashionViewer.capture), because headless screenshots do not see
// WebGPU canvases. The run fails unless the renderer reports the WebGPU
// backend, so a WebGL2 fallback can never be passed off as a WebGPU render.
//
// Browser: CHROMIUM_PATH if set, otherwise the installed Google Chrome channel.
// The hardware GPU is tried first; if WebGPU is unavailable or the device is
// lost, the capture is retried on SwiftShader (software Vulkan), which is how
// it runs on GPU-less CI machines. The output JSON names the adapter used.
// Needs `npm install` at the repository root (playwright-core).

import { createServer } from "node:http";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".png": "image/png", ".f32": "application/octet-stream",
  ".u32": "application/octet-stream",
};

function parseArgs(argv) {
  const args = { scene: undefined, out: "outputs/renders", phases: [0.5], yaws: [0], width: 1280, height: 960, timeoutMs: 180_000 };
  for (let i = 0; i < argv.length; i += 2) {
    const [key, value] = [argv[i], argv[i + 1]];
    if (key === "--scene") args.scene = value;
    else if (key === "--out") args.out = value;
    else if (key === "--phases") args.phases = value.split(",").map(Number);
    else if (key === "--yaws") args.yaws = value.split(",").map(Number);
    else if (key === "--width") args.width = Number(value);
    else if (key === "--height") args.height = Number(value);
    else if (key === "--timeout-ms") args.timeoutMs = Number(value);
    else throw new Error(`unknown argument ${key}`);
  }
  if (!args.scene || !/^[A-Za-z0-9_.-]+$/.test(args.scene)) throw new Error("--scene <name> is required (a folder in web/scenes)");
  return args;
}

function serveRepository() {
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://localhost");
      const file = path.resolve(ROOT, "." + decodeURIComponent(url.pathname));
      if (!file.startsWith(ROOT + path.sep) || !(await stat(file)).isFile()) throw new Error("not found");
      response.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" });
      response.end(await readFile(file));
    } catch {
      response.writeHead(404).end();
    }
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

const GPU_MODES = {
  hardware: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist"],
  // Without the explicit SwiftShader Vulkan + ANGLE pair, Chromium's software
  // WebGPU device is lost ~0.1 s after creation ("A valid external Instance
  // reference no longer exists").
  software: [
    "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader",
    "--use-angle=swiftshader", "--disable-vulkan-surface", "--ignore-gpu-blocklist",
  ],
};

async function captureWith(mode, args, port) {
  const browser = await chromium.launch({
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : { channel: "chrome" }),
    args: GPU_MODES[mode],
  });
  try {
    const page = await browser.newPage({ viewport: { width: args.width, height: args.height } });
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    if (mode === "hardware") {
      // Skip straight to the software path when there is no real GPU adapter,
      // instead of loading the scene only to lose the device.
      await page.goto(`http://127.0.0.1:${port}/package.json`);
      const probe = await page.evaluate(async () => {
        const adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
        const info = adapter?.info ?? {};
        return { ok: Boolean(adapter), fallback: Boolean(info.isFallbackAdapter), description: `${info.vendor ?? ""} ${info.description ?? ""}` };
      });
      if (!probe.ok || probe.fallback || /swiftshader/i.test(probe.description)) {
        throw new Error(`hardware: no hardware WebGPU adapter (${JSON.stringify(probe)})`);
      }
    }
    await page.goto(`http://127.0.0.1:${port}/anny-cloth-webgpu.html?scene=${encodeURIComponent(args.scene)}&view=front`);
    await page.waitForFunction(
      () => window.__fashionViewer || document.querySelector("#error:not([hidden])"),
      null,
      { timeout: args.timeoutMs },
    );
    // Give asynchronous pipeline compilation a moment so the frames include every mesh.
    await page.waitForTimeout(3000);
    const state = await page.evaluate(() => ({
      backend: window.__fashionViewer?.backend ?? null,
      deviceLost: window.__fashionViewer?.deviceLost ?? null,
      error: document.querySelector("#error:not([hidden])")?.textContent ?? null,
    }));
    if (state.backend !== "webgpu" || state.deviceLost) {
      throw new Error(`${mode}: viewer not running on WebGPU (${JSON.stringify(state)}) ${pageErrors.join("; ")}`);
    }
    await mkdir(path.resolve(ROOT, args.out), { recursive: true });
    const written = [];
    for (const phase of args.phases) {
      for (const yawDegrees of args.yaws) {
        const dataUrl = await page.evaluate((options) => window.__fashionViewer.capture(options), { phase, yawDegrees });
        const file = path.resolve(ROOT, args.out, `${args.scene}_p${phase.toFixed(2)}_y${yawDegrees}.png`);
        await writeFile(file, Buffer.from(dataUrl.split(",")[1], "base64"));
        written.push(path.relative(ROOT, file));
      }
    }
    const adapter = await page.evaluate(async () => {
      const info = (await navigator.gpu.requestAdapter())?.info ?? {};
      return [info.vendor, info.architecture, info.description].filter(Boolean).join(" ");
    });
    return { scene: args.scene, backend: state.backend, gpuMode: mode, adapter, frames: written };
  } finally {
    await browser.close();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await stat(path.join(ROOT, "web", "scenes", args.scene, "scene.json"));
  const server = await serveRepository();
  const { port } = server.address();
  const failures = [];
  try {
    for (const mode of process.env.WEBGPU_SOFTWARE === "1" ? ["software"] : ["hardware", "software"]) {
      try {
        console.log(JSON.stringify(await captureWith(mode, args, port), null, 2));
        return;
      } catch (error) {
        failures.push(error.message);
      }
    }
    throw new Error(failures.join("\n"));
  } finally {
    server.close();
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
