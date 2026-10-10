import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const query = new URLSearchParams(window.location.search);
// The default must be a scene committed under web/scenes/.
const sceneName = query.get("scene") || "anny-skirt-walk-one-step-2s-contactfix-v10-webgpu";
const SCENE_ROOT = `./web/scenes/${sceneName}/`;
const canvas = document.querySelector("#viewport");
const loading = document.querySelector("#loading");
const errorBox = document.querySelector("#error");
const backendBadge = document.querySelector("#backend");
const playButton = document.querySelector("#play");
const scrubber = document.querySelector("#scrubber");
const speedControl = document.querySelector("#speed");
const timecode = document.querySelector("#timecode");

const loadJson = async (name) => {
  const response = await fetch(SCENE_ROOT + name);
  if (!response.ok) throw new Error(`Unable to load ${name}: ${response.status}`);
  return response.json();
};

const loadTyped = async (name, Type) => {
  const response = await fetch(SCENE_ROOT + name);
  if (!response.ok) throw new Error(`Unable to load ${name}: ${response.status}`);
  return new Type(await response.arrayBuffer());
};

const hermiteSample = (positions, velocities, vertexCount, times, phase, output) => {
  const scaled = phase * (times.length - 1);
  const frame = Math.min(Math.floor(scaled), times.length - 2);
  const u = scaled - frame;
  const h = times[frame + 1] - times[frame];
  const h00 = 2 * u * u * u - 3 * u * u + 1;
  const h10 = u * u * u - 2 * u * u + u;
  const h01 = -2 * u * u * u + 3 * u * u;
  const h11 = u * u * u - u * u;
  const stride = vertexCount * 3;
  const a = frame * stride;
  const b = (frame + 1) * stride;
  for (let i = 0; i < stride; i += 1) {
    output[i] = h00 * positions[a + i] + h10 * h * velocities[a + i]
      + h01 * positions[b + i] + h11 * h * velocities[b + i];
  }
};

const linearSample = (positions, vertexCount, frameCount, phase, output) => {
  const scaled = phase * (frameCount - 1);
  const frame = Math.min(Math.floor(scaled), frameCount - 2);
  const u = scaled - frame;
  const stride = vertexCount * 3;
  const a = frame * stride;
  const b = (frame + 1) * stride;
  for (let i = 0; i < stride; i += 1) {
    output[i] = positions[a + i] * (1 - u) + positions[b + i] * u;
  }
};

const textureLoader = new THREE.TextureLoader();
const loadTexture = async (name) => {
  const texture = await textureLoader.loadAsync(SCENE_ROOT + name);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
};

try {
  const manifest = await loadJson("scene.json");
  const f = manifest.files;
  const [
    bodyPositions, bodyIndices,
    shirtPositions, shirtVelocities, shirtIndices, shirtUvs,
    skirtPositions, skirtVelocities, skirtIndices, skirtUvs,
    shirtTexture, skirtTexture,
  ] = await Promise.all([
    loadTyped(f.bodyPositions, Float32Array),
    loadTyped(f.bodyIndices, Uint32Array),
    loadTyped(f.shirtPositions, Float32Array),
    loadTyped(f.shirtVelocities, Float32Array),
    loadTyped(f.shirtIndices, Uint32Array),
    loadTyped(f.shirtUvs, Float32Array),
    loadTyped(f.skirtPositions, Float32Array),
    loadTyped(f.skirtVelocities, Float32Array),
    loadTyped(f.skirtIndices, Uint32Array),
    loadTyped(f.skirtUvs, Float32Array),
    loadTexture(f.shirtTexture),
    loadTexture(f.skirtTexture),
  ]);

  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  let deviceLost = null;
  renderer.onDeviceLost = (info) => {
    deviceLost = info?.message || "unknown";
    errorBox.hidden = false;
    errorBox.textContent = `WebGPU device lost: ${deviceLost}`;
    backendBadge.textContent = "GPU DEVICE LOST";
  };
  await renderer.init();

  const usingWebGPU = renderer.backend?.isWebGPUBackend === true;
  backendBadge.textContent = usingWebGPU ? "WEBGPU ACTIVE" : "WEBGL2 FALLBACK";
  backendBadge.classList.toggle("webgpu", usingWebGPU);
  document.documentElement.dataset.renderer = usingWebGPU ? "webgpu" : "webgl2";

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe9ecec);
  scene.fog = new THREE.Fog(0xe9ecec, 3.0, 5.5);

  const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.02, 20);
  const minimum = new THREE.Vector3().fromArray(manifest.bounds.minimum);
  const maximum = new THREE.Vector3().fromArray(manifest.bounds.maximum);
  const center = minimum.clone().add(maximum).multiplyScalar(0.5);
  const size = maximum.clone().sub(minimum);
  const radius = Math.max(size.x, size.y, size.z) * 0.62;
  const frontView = query.get("view") === "front";
  camera.position.set(
    center.x + (frontView ? 0 : radius * 1.55),
    center.y + radius * 0.25,
    center.z + (frontView ? radius * 2.95 : radius * 2.65)
  );
  camera.lookAt(center);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center).add(new THREE.Vector3(0, -0.03, 0));
  controls.enableDamping = true;
  controls.minDistance = radius * 1.2;
  controls.maxDistance = radius * 6;
  controls.update();

  scene.add(new THREE.HemisphereLight(0xf8fbfa, 0x88817b, 2.1));
  const key = new THREE.DirectionalLight(0xffffff, 3.5);
  key.position.set(center.x + 1.7, center.y + 2.2, center.z + 2.4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xcbd6d3, 1.5);
  rim.position.set(center.x - 2, center.y + 0.5, center.z - 1.5);
  scene.add(rim);

  const makeGeometry = (vertexCount, positions, indices, uvs = null) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    if (uvs) geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    if (geometry.getAttribute("position").count !== vertexCount) throw new Error("Vertex count mismatch");
    return geometry;
  };

  // Display-only intrinsic smoothing: preserve garment boundaries while
  // removing the faceted look caused by the simulation resolution.
  const makeInteriorSmoother = (indices, vertexCount, strength = 0.1) => {
    const neighbors = Array.from({ length: vertexCount }, () => new Set());
    const edgeUse = new Map();
    for (let i = 0; i < indices.length; i += 3) {
      const tri = [indices[i], indices[i + 1], indices[i + 2]];
      for (let e = 0; e < 3; e += 1) {
        const a = tri[e];
        const b = tri[(e + 1) % 3];
        neighbors[a].add(b);
        neighbors[b].add(a);
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        edgeUse.set(key, (edgeUse.get(key) || 0) + 1);
      }
    }
    const boundary = new Uint8Array(vertexCount);
    for (const key of edgeUse.keys()) {
      if (edgeUse.get(key) === 1) {
        const [a, b] = key.split(":").map(Number);
        boundary[a] = 1;
        boundary[b] = 1;
      }
    }
    return (positions) => {
      const next = positions.slice();
      for (let v = 0; v < vertexCount; v += 1) {
        if (boundary[v] || neighbors[v].size < 3) continue;
        const average = [0, 0, 0];
        for (const n of neighbors[v]) {
          average[0] += positions[3 * n];
          average[1] += positions[3 * n + 1];
          average[2] += positions[3 * n + 2];
        }
        const inv = 1 / neighbors[v].size;
        next[3 * v] += strength * (average[0] * inv - positions[3 * v]);
        next[3 * v + 1] += strength * (average[1] * inv - positions[3 * v + 1]);
        next[3 * v + 2] += strength * (average[2] * inv - positions[3 * v + 2]);
      }
      positions.set(next);
    };
  };

  const bodyNow = bodyPositions.slice(0, manifest.meshes.body.vertexCount * 3);
  const shirtNow = shirtPositions.slice(0, manifest.meshes.shirt.vertexCount * 3);
  const skirtNow = skirtPositions.slice(0, manifest.meshes.skirt.vertexCount * 3);
  const bodyGeometry = makeGeometry(manifest.meshes.body.vertexCount, bodyNow, bodyIndices);
  const shirtGeometry = makeGeometry(manifest.meshes.shirt.vertexCount, shirtNow, shirtIndices, shirtUvs);
  const skirtGeometry = makeGeometry(manifest.meshes.skirt.vertexCount, skirtNow, skirtIndices, skirtUvs);
  const smoothShirtDisplay = makeInteriorSmoother(shirtIndices, manifest.meshes.shirt.vertexCount, 0.035);
  // Keep the physical shirt fully dynamic, but use a smoother display shell
  // for the printed front faces so a low-resolution fold cannot shear the
  // graphic into a diagonal stack of triangles.
  const smoothSkirtDisplay = makeInteriorSmoother(skirtIndices, manifest.meshes.skirt.vertexCount, 0.055);
  manifest.meshes.shirt.groups.forEach((group) => shirtGeometry.addGroup(group.start, group.count, group.materialIndex));

  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xd4aa8c, roughness: 0.9, metalness: 0 });
  const clothDepthBias = {
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  };
  // The print is a material on the physical shirt faces, rather than a
  // coplanar overlay. A separate decal mesh z-fights with the moving cloth
  // shell in WebGPU and makes the graphic flicker or disappear on folds.
  shirtTexture.wrapS = THREE.ClampToEdgeWrapping;
  shirtTexture.wrapT = THREE.ClampToEdgeWrapping;
  shirtTexture.minFilter = THREE.LinearMipmapLinearFilter;
  shirtTexture.magFilter = THREE.LinearFilter;
  shirtTexture.needsUpdate = true;
  const printMaterial = new THREE.MeshStandardMaterial({
    map: shirtTexture,
    color: 0xffffff,
    roughness: 0.94,
    // The dog belongs to the outward chest surface only. Rendering this
    // group double-sided made the same texture visible from the back.
    side: THREE.FrontSide,
    ...clothDepthBias,
  });
  const shirtMaterial = new THREE.MeshStandardMaterial({ color: 0xf6f6f2, roughness: 0.94, side: THREE.DoubleSide, ...clothDepthBias });
  skirtTexture.wrapS = skirtTexture.wrapT = THREE.RepeatWrapping;
  skirtTexture.repeat.set(3, 1);
  const skirtMaterial = new THREE.MeshStandardMaterial({ map: skirtTexture, color: 0xffffff, roughness: 0.86, side: THREE.DoubleSide, ...clothDepthBias });

  const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
  const shirtMesh = new THREE.Mesh(shirtGeometry, [printMaterial, shirtMaterial]);
  const skirtMesh = new THREE.Mesh(skirtGeometry, skirtMaterial);
  bodyMesh.castShadow = bodyMesh.receiveShadow = true;
  shirtMesh.castShadow = shirtMesh.receiveShadow = true;
  skirtMesh.castShadow = skirtMesh.receiveShadow = true;
  scene.add(bodyMesh, shirtMesh, skirtMesh);

  const groundY = minimum.y - 0.012;
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(radius * 2.8, 96),
    new THREE.MeshStandardMaterial({ color: 0xd8dcdb, roughness: 1.0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(center.x, groundY, center.z);
  ground.receiveShadow = true;
  scene.add(ground);

  const audit = manifest.physicsAudit;
  document.querySelector("#body-gap").textContent = `${(audit.minimumBodyGapMeters * 1e3).toFixed(3)} mm`;
  document.querySelector("#metric-edges").textContent = audit.metricProjectionEnabled
    ? `${audit.skirtEdgeCount - audit.skirtViolatingEdges}/${audit.skirtEdgeCount}`
    : "OFF";
  document.querySelector("#lie-steps").textContent = String(audit.lieMessageSubsteps);

  let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let phase = 0;
  let previousTime = 0;
  let speed = 1;
  const physicalDuration = manifest.physicalDurationSeconds;
  const displayDuration = manifest.displayLoopSeconds;

  const updateField = (nextPhase) => {
    phase = Math.min(Math.max(nextPhase, 0), 1);
    linearSample(bodyPositions, manifest.meshes.body.vertexCount, manifest.frameCount, phase, bodyNow);
    // The accepted physical knots already satisfy the SDF contact pass.
    // Linear sampling cannot overshoot those collision-safe states, whereas
    // cubic Hermite interpolation can create a transient shoulder/waist
    // penetration between knots when velocities change at an event.
    linearSample(shirtPositions, manifest.meshes.shirt.vertexCount, manifest.frameCount, phase, shirtNow);
    linearSample(skirtPositions, manifest.meshes.skirt.vertexCount, manifest.frameCount, phase, skirtNow);
    smoothShirtDisplay(shirtNow);
    smoothSkirtDisplay(skirtNow);
    for (const geometry of [bodyGeometry, shirtGeometry, skirtGeometry]) {
      geometry.getAttribute("position").needsUpdate = true;
      geometry.computeVertexNormals();
    }
    scrubber.value = String(phase);
    timecode.value = `${(phase * physicalDuration).toFixed(3)} s`;
  };

  playButton.addEventListener("click", () => {
    playing = !playing;
    playButton.textContent = playing ? "II" : ">";
    playButton.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
  });
  scrubber.addEventListener("input", () => {
    playing = false;
    playButton.textContent = ">";
    updateField(Number(scrubber.value));
  });
  speedControl.addEventListener("change", () => { speed = Number(speedControl.value); });

  const animate = (timestamp) => {
    const delta = previousTime ? Math.min((timestamp - previousTime) / 1000, 0.05) : 0;
    previousTime = timestamp;
    if (playing) updateField((phase + delta * speed / displayDuration) % 1);
    controls.update();
    renderer.render(scene, camera);
  };
  updateField(0);
  loading.hidden = true;
  renderer.setAnimationLoop(animate);

  // Deterministic frame capture for scripts/capture_webgpu.mjs and the agent's
  // render_outfit tool: pause, set the physical phase and an orbit yaw, render
  // once and read the WebGPU canvas back (headless screenshots cannot see it).
  window.__fashionViewer = {
    backend: usingWebGPU ? "webgpu" : "webgl2",
    get deviceLost() {
      return deviceLost;
    },
    frameCount: manifest.frameCount,
    async capture({ phase: capturePhase = 0.5, yawDegrees = 0 } = {}) {
      playing = false;
      renderer.setAnimationLoop(null);
      updateField(capturePhase);
      const offset = camera.position.clone().sub(controls.target);
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), (yawDegrees * Math.PI) / 180);
      camera.position.copy(controls.target).add(offset);
      camera.lookAt(controls.target);
      renderer.render(scene, camera);
      if (deviceLost) throw new Error(`WebGPU device lost: ${deviceLost}`);
      const image = canvas.toDataURL("image/png");
      camera.position.copy(controls.target).add(offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), (-yawDegrees * Math.PI) / 180));
      return image;
    },
  };

  if (query.get("record") === "1" && typeof canvas.captureStream === "function") {
    const stream = canvas.captureStream(30);
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("stop", async () => {
      const blob = new Blob(chunks, { type: mimeType });
      const endpoint = query.get("recordEndpoint")
        || `http://127.0.0.1:8770/capture/${encodeURIComponent(sceneName)}.webm`;
      let saved = false;
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": mimeType },
          body: blob,
        });
        saved = response.ok;
      } catch (_error) {
        saved = false;
      }
      if (!saved) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `neuralclothsim-${sceneName}.webm`;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      }
      backendBadge.textContent = saved ? "CAPTURE SAVED" : "CAPTURE DOWNLOADED";
    });
    phase = 0;
    previousTime = 0;
    playing = true;
    recorder.start(250);
    window.setTimeout(() => recorder.stop(), Math.max(displayDuration, 0.1) * 1000 + 250);
  }

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
} catch (error) {
  console.error(error);
  loading.hidden = true;
  errorBox.hidden = false;
  errorBox.textContent = `WebGPU scene failed: ${error.message}`;
  backendBadge.textContent = "RENDER ERROR";
}
