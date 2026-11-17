import * as THREE from "three";
import { OrbitControls } from "./vendor/controls/OrbitControls.js";
import { GLTFLoader } from "./vendor/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const runsEl = $("runs");
const veil = $("veil");
const veilLabel = document.querySelector("#veil .v-label");
const veilFill = document.querySelector("#veil .v-fill");
const veilPct = document.querySelector("#veil .v-pct");
const errorEl = $("error");
const plate = $("plate");

// ---------- renderer ----------

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setClearColor(0x000000, 0); // 背景交给 CSS 渐变
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// 程序化摄影棚环境（无外部 HDR）：顶柔光箱 + 冷补光 + 暖轮廓光
function buildEnvironment() {
  const env = new THREE.Scene();
  env.background = new THREE.Color(0x14161c);
  const panel = (color, intensity, w, h, pos) => {
    const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    mat.color.set(color).multiplyScalar(intensity);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    mesh.position.set(...pos);
    mesh.lookAt(0, 0, 0);
    env.add(mesh);
  };
  panel(0xffffff, 9, 6, 3, [0, 6, 0]);
  panel(0xcfe0ff, 3.5, 4, 4, [-6, 2, 2]);
  panel(0xffe2c4, 3, 4, 4, [6, 1, -2]);
  panel(0xffffff, 1.5, 10, 4, [0, 1, -8]);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(env, 0.04).texture;
  pmrem.dispose();
  return texture;
}
scene.environment = buildEnvironment();

scene.add(new THREE.HemisphereLight(0xdde6ff, 0x1c1a18, 0.45));

const keyLight = new THREE.DirectionalLight(0xfff2e2, 1.7);
keyLight.position.set(1.8, 3.2, 2.4);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.bias = -0.0002;
keyLight.shadow.radius = 6;
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0xbfd2ff, 1.0);
rimLight.position.set(-2.4, 2.0, -2.6);
scene.add(rimLight);

// 地面：柔阴影承接盘（默认）+ 可选参考网格
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(3.2, 64),
  new THREE.ShadowMaterial({ opacity: 0.5, color: 0x000000 }),
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// 底盘微光环，展台感
const disc = new THREE.Mesh(
  new THREE.RingGeometry(1.02, 1.06, 96),
  new THREE.MeshBasicMaterial({ color: 0xc9a876, transparent: true, opacity: 0.28,
                                side: THREE.DoubleSide }),
);
disc.rotation.x = -Math.PI / 2;
scene.add(disc);

const grid = new THREE.GridHelper(4, 40, 0x3a3d4e, 0x232530);
grid.visible = false;
scene.add(grid);

// ---------- camera / controls ----------

const camera = new THREE.PerspectiveCamera(32, 1, 0.001, 200);
camera.position.set(1.2, 1.3, 2.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.zoomToCursor = false;
controls.screenSpacePanning = true; // 屏幕系平移：垂直拖动可从脚移到头
controls.autoRotate = true;
controls.autoRotateSpeed = 0.9;
controls.target.set(0, 0.9, 0);
controls.update();

const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let current = null; // { root, vrm, skeletonHelper, run, model }
let loadToken = 0;
let runs = [];
let camTween = null; // { fromPos,toPos,fromTgt,toTgt,t,dur }

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function showError(text) {
  errorEl.textContent = text;
  errorEl.style.display = "block";
}

function setProgress(label, frac) {
  veilLabel.textContent = label;
  if (frac == null) {
    veilPct.textContent = "";
    veilFill.style.width = "40%";
  } else {
    veilPct.textContent = `${Math.round(frac * 100)}%`;
    veilFill.style.width = `${Math.round(frac * 100)}%`;
  }
}

// ---------- framing & camera tween ----------

function computeFrame(root) {
  const box = new THREE.Box3().setFromObject(root);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 0.01);

  camera.near = Math.max(radius / 500, 0.0002); // 发丝级特写不裁剪
  camera.far = radius * 200;
  camera.updateProjectionMatrix();
  controls.minDistance = radius * 0.04;
  controls.maxDistance = radius * 40;

  const distance = (radius / Math.tan(camera.fov * Math.PI / 360)) * 1.45;
  ground.position.y = box.min.y;
  disc.position.y = box.min.y + 0.002;
  const discScale = radius / 1.0;
  disc.scale.setScalar(discScale);
  grid.position.y = box.min.y;
  keyLight.shadow.camera.left = -radius * 1.5;
  keyLight.shadow.camera.right = radius * 1.5;
  keyLight.shadow.camera.top = radius * 1.5;
  keyLight.shadow.camera.bottom = -radius * 1.5;
  keyLight.shadow.camera.far = radius * 20;
  keyLight.shadow.camera.updateProjectionMatrix();
  return { center: sphere.center.clone(), radius, distance };
}

function flyTo(center, distance, dir) {
  const toPos = center.clone().addScaledVector(dir, distance);
  camTween = {
    fromPos: camera.position.clone(), toPos,
    fromTgt: controls.target.clone(), toTgt: center.clone(),
    t: 0, dur: 0.9,
  };
}

function frameObject(root, animate) {
  const frame = computeFrame(root);
  const dir = camera.position.clone().sub(controls.target);
  if (dir.lengthSq() < 1e-6) dir.set(0.35, 0.22, 1);
  dir.normalize();
  if (animate) {
    flyTo(frame.center, frame.distance, dir);
  } else {
    camTween = null;
    controls.target.copy(frame.center);
    camera.position.copy(frame.center).addScaledVector(dir, frame.distance);
    controls.update();
  }
}

// ---------- model lifecycle ----------

function disposeCurrent() {
  if (!current) return;
  scene.remove(current.root);
  if (current.skeletonHelper) {
    scene.remove(current.skeletonHelper);
    current.skeletonHelper.dispose();
  }
  VRMUtils.deepDispose(current.root);
  current = null;
}

function prepareMaterials(root) {
  root.traverse((node) => {
    if (!node.isMesh) return;
    node.castShadow = true;
    node.receiveShadow = false;
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    for (const m of mats) {
      if (!m) continue;
      m.userData.baseOpacity = m.opacity;
      for (const value of Object.values(m)) {
        if (value?.isTexture) value.anisotropy = maxAnisotropy; // 斜视角发丝不糊
      }
    }
  });
}

function setModelOpacity(root, a) {
  root.traverse((node) => {
    if (!node.isMesh) return;
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    for (const m of mats) {
      if (!m) continue;
      const base = m.userData.baseOpacity ?? 1;
      m.transparent = a < 1 || base < 1;
      m.opacity = base * a;
    }
  });
}

let fadeIn = null; // { t, dur }

function applyWireframe(enabled) {
  if (!current) return;
  current.root.traverse((node) => {
    if (!node.isMesh) return;
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    for (const m of mats) if (m) m.wireframe = enabled;
  });
}

function applySkeleton(enabled) {
  if (!current) return;
  if (enabled && !current.skeletonHelper) {
    current.skeletonHelper = new THREE.SkeletonHelper(current.root);
    scene.add(current.skeletonHelper);
  } else if (!enabled && current.skeletonHelper) {
    scene.remove(current.skeletonHelper);
    current.skeletonHelper.dispose();
    current.skeletonHelper = null;
  }
}

// ---------- 铭牌 ----------

const fmtNum = (n) => n.toLocaleString("en-US");

function modelStats(root) {
  let tris = 0, verts = 0, meshes = 0, bones = 0;
  root.traverse((node) => {
    if (node.isMesh && node.geometry) {
      meshes += 1;
      const pos = node.geometry.getAttribute("position");
      if (pos) verts += pos.count;
      const idx = node.geometry.getIndex();
      tris += Math.round((idx ? idx.count : (pos ? pos.count : 0)) / 3);
    } else if (node.isBone) bones += 1;
  });
  return { tris, verts, meshes, bones };
}

function renderPlate(run, model, stats, vrm) {
  const entry = runs.find((r) => r.run === run);
  const reports = entry?.reports ?? {};
  const niceName = model.name.replace(/\.(glb|vrm)$/i, "").replaceAll("_", " ");
  plate.querySelector(".p-run").textContent = run;
  plate.querySelector(".p-name").textContent = niceName;
  plate.querySelector(".p-type").textContent =
    vrm ? "VRM 1.0 · 虚拟人格式" : "GLB · glTF 二进制";

  // 全部走 DOM/textContent：报告字段与文件名一样不可信
  const chipsEl = plate.querySelector(".chips");
  chipsEl.replaceChildren();
  const chip = (num, unit) => {
    const c = document.createElement("span");
    c.className = "chip";
    const b = document.createElement("b");
    b.textContent = num;
    c.append(b, ` ${unit}`);
    chipsEl.appendChild(c);
  };
  chip(fmtNum(stats.tris), "tris");
  chip(fmtNum(stats.verts), "verts");
  if (stats.bones) chip(String(stats.bones), "bones");
  chip((model.size / 1048576).toFixed(1), "MB");

  const reportEl = plate.querySelector(".p-report");
  reportEl.replaceChildren();
  const line = (key, text) => {
    if (reportEl.firstChild) reportEl.appendChild(document.createElement("br"));
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = key;
    reportEl.append(k, ` ${text}`);
  };
  const gen = reports.generation;
  if (gen?.ckpt) {
    line("生成", String(gen.ckpt) +
      (gen.seed != null ? ` · seed ${gen.seed}` : "") +
      (gen.octree_resolution ? ` · octree ${gen.octree_resolution}` : ""));
  }
  const clean = reports.cleanup;
  if (clean?.output_faces != null) {
    line("清洗", `${fmtNum(clean.output_faces)} faces` +
      (clean.input_faces ? `（原 ${fmtNum(clean.input_faces)}）` : "") +
      (clean.method ? ` · ${clean.method}` : ""));
  }
  const rig = reports.rig;
  if (rig?.bones != null) {
    line("绑骨", `${rig.bones} bones` +
      (rig.weighted_ratio != null
        ? ` · 权重覆盖 ${(rig.weighted_ratio * 100).toFixed(0)}%` : "") +
      (rig.bind_method ? ` · ${rig.bind_method}` : ""));
  }

  const gateEl = plate.querySelector(".p-gate");
  const gate = reports.vrm_gate;
  gateEl.replaceChildren();
  if (gate) {
    gateEl.className = `p-gate ${gate.all_pass ? "pass" : "fail"}`;
    gateEl.textContent = gate.all_pass
      ? `✓ VRM 门检全过（${gate.passed}/${gate.total}）`
      : `⚠ VRM 门检 ${gate.passed}/${gate.total}`;
    for (const b of gate.blockers ?? []) {
      const div = document.createElement("div");
      div.className = "blocker";
      div.textContent = `· ${b}`;
      gateEl.appendChild(div);
    }
  } else {
    gateEl.className = "p-gate";
    gateEl.textContent = "";
  }
  plate.classList.add("on");
}

// ---------- loading ----------

async function fetchWithProgress(url, totalSize, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setProgress(label, totalSize ? received / totalSize : null);
  }
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
  return buffer.buffer;
}

async function loadModel(run, model) {
  const myToken = ++loadToken; // 并发加载竞态防护：后发的请求赢
  const label = `${run} / ${model.name}`;
  errorEl.style.display = "none";
  plate.classList.remove("on");
  // 切换：先淡入 veil，再换模型，再淡出 —— 平滑过渡
  veil.classList.remove("off");
  setProgress(label, 0);
  await new Promise((r) => setTimeout(r, current ? 260 : 0));
  if (myToken !== loadToken) return;
  let gltf;
  try {
    const url = `/api/file?run=${encodeURIComponent(run)}&f=${encodeURIComponent(model.name)}`;
    const data = await fetchWithProgress(url, model.size, label);
    if (myToken !== loadToken) return;
    setProgress(`装配 ${model.name}`, null);
    gltf = await loader.parseAsync(data, "");
  } catch (err) {
    if (myToken !== loadToken) return;
    console.error(err);
    veil.classList.add("off");
    showError(`载入失败 ${label}: ${err.message || err}`);
    return;
  }
  const vrm = gltf.userData.vrm || null;
  const root = vrm ? vrm.scene : gltf.scene;
  if (myToken !== loadToken) { // 解析完成时已被取代：丢弃并释放，不留孤儿
    VRMUtils.deepDispose(root);
    return;
  }
  disposeCurrent(); // 只在新模型就绪后才释放旧模型
  if (vrm) VRMUtils.rotateVRM0(vrm);
  prepareMaterials(root);
  setModelOpacity(root, 0);
  scene.add(root);
  current = { root, vrm, skeletonHelper: null, run, model };
  frameObject(root, true);
  applyWireframe($("opt-wire").checked);
  applySkeleton($("opt-bones").checked);
  fadeIn = { t: 0, dur: 0.7 };
  renderPlate(run, model, modelStats(root), vrm);
  veil.classList.add("off");
  document.body.classList.add("ready");
  for (const btn of runsEl.querySelectorAll(".piece")) {
    btn.classList.toggle("active",
      btn.dataset.file === model.name && btn.dataset.run === run);
  }
}

// ---------- catalog UI ----------

function preferredOrder(models) {
  const rank = (m) => (m.name.endsWith(".vrm") ? 0 : m.name.includes("rigged") ? 1 : 2);
  return [...models].sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
}

function buildRunList() {
  runsEl.textContent = "";
  let first = null;
  for (const entry of runs) {
    const block = document.createElement("div");
    block.className = "run-block";
    const gate = entry.reports?.vrm_gate;
    const head = document.createElement("div");
    head.className = "run-head";
    head.replaceChildren(); // 注入防护：目录名只走 textContent
    const nameEl = document.createElement("span");
    nameEl.className = "name";
    nameEl.textContent = entry.run;
    head.appendChild(nameEl);
    if (gate) {
      const pill = document.createElement("span");
      pill.className = `gate-pill ${gate.all_pass ? "pass" : "fail"}`;
      pill.textContent = `GATE ${gate.passed}/${gate.total}`;
      head.appendChild(pill);
    }
    block.appendChild(head);
    for (const model of preferredOrder(entry.models)) {
      const btn = document.createElement("button");
      btn.className = "piece";
      btn.dataset.run = entry.run;
      btn.dataset.file = model.name;
      const ext = model.name.endsWith(".vrm") ? "vrm" : "glb";
      const short = model.name.replace(/\.(glb|vrm)$/i, "");
      // 注入防护：文件名只走 textContent / setAttribute，不进 innerHTML
      const row1 = document.createElement("div");
      row1.className = "row1";
      const tag = document.createElement("span");
      tag.className = `tag ${ext}`;
      tag.textContent = ext.toUpperCase();
      const fname = document.createElement("span");
      fname.className = "fname";
      fname.setAttribute("title", model.name);
      fname.textContent = short;
      row1.append(tag, fname);
      const row2 = document.createElement("div");
      row2.className = "row2";
      const sizeEl = document.createElement("span");
      sizeEl.textContent = `${(model.size / 1048576).toFixed(1)} MB`;
      row2.appendChild(sizeEl);
      btn.append(row1, row2);
      btn.addEventListener("click", () => loadModel(entry.run, model));
      block.appendChild(btn);
      if (!first) first = { run: entry.run, model };
    }
    runsEl.appendChild(block);
  }
  $("run-count").textContent = `${runs.length} 组产出`;
  return first;
}

async function init() {
  let first = null;
  try {
    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    runs = (await response.json()).runs;
    first = buildRunList();
  } catch (err) {
    showError(`无法列出 runs: ${err.message || err}`);
    veil.classList.add("off");
    return;
  }
  if (!first) {
    showError("3d/chargen/runs 下没有 GLB/VRM");
    veil.classList.add("off");
    return;
  }
  loadModel(first.run, first.model);
}

// ---------- events ----------

$("opt-rotate").addEventListener("change", (e) => {
  controls.autoRotate = e.target.checked;
});
$("opt-grid").addEventListener("change", (e) => { grid.visible = e.target.checked; });
$("opt-wire").addEventListener("change", (e) => applyWireframe(e.target.checked));
$("opt-bones").addEventListener("change", (e) => applySkeleton(e.target.checked));

$("collapse").addEventListener("click", () => {
  $("catalog").classList.add("hidden");
  $("expand-btn").classList.add("show");
});
$("expand-btn").addEventListener("click", () => {
  $("catalog").classList.remove("hidden");
  $("expand-btn").classList.remove("show");
});

const raycaster = new THREE.Raycaster();
renderer.domElement.addEventListener("dblclick", (event) => {
  if (!current) return;
  const ndc = new THREE.Vector2(
    (event.clientX / window.innerWidth) * 2 - 1,
    -(event.clientY / window.innerHeight) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObject(current.root, true);
  if (hits.length) {
    const from = controls.target.clone();
    camTween = { fromPos: camera.position.clone(), toPos: camera.position.clone(),
                 fromTgt: from, toTgt: hits[0].point.clone(), t: 0, dur: 0.4 };
  }
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  switch (event.key.toLowerCase()) {
    case "f":
      if (current) frameObject(current.root, true);
      break;
    case " ": {
      event.preventDefault();
      const box = $("opt-rotate");
      box.checked = !box.checked;
      controls.autoRotate = box.checked;
      break;
    }
  }
});

controls.addEventListener("start", () => { camTween = null; controls.autoRotate = false; });
controls.addEventListener("end", () => {
  controls.autoRotate = $("opt-rotate").checked;
});

// ---------- main loop ----------

const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2);
let lastFrame = performance.now(); // 不用已弃用的 THREE.Clock
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const delta = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;
  if (current?.vrm) current.vrm.update(delta);
  if (camTween) {
    camTween.t += delta / camTween.dur;
    const k = easeInOut(Math.min(camTween.t, 1));
    camera.position.lerpVectors(camTween.fromPos, camTween.toPos, k);
    controls.target.lerpVectors(camTween.fromTgt, camTween.toTgt, k);
    if (camTween.t >= 1) camTween = null;
  }
  if (fadeIn && current) {
    fadeIn.t += delta / fadeIn.dur;
    const k = Math.min(fadeIn.t, 1);
    setModelOpacity(current.root, k);
    if (k >= 1) fadeIn = null;
  }
  if (camTween) {
    camera.lookAt(controls.target);   // tween 期间绕开 controls，避免阻尼状态打架
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
});

init();

// 调试/测试句柄
window.__showcase = {
  scene, camera, controls,
  get current() { return current; },
  box() {
    if (!current) return null;
    const b = new THREE.Box3().setFromObject(current.root);
    return { min: b.min.toArray(), max: b.max.toArray() };
  },
};
