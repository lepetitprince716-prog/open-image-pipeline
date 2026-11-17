import * as THREE from "three";
import { OrbitControls } from "./vendor/controls/OrbitControls.js";
import { GLTFLoader } from "./vendor/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const stage = document.getElementById("stage");
const statusEl = document.getElementById("status");
const runSelect = document.getElementById("run-select");
const modelList = document.getElementById("model-list");
const reportEl = document.getElementById("report");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14141a);
scene.add(new THREE.GridHelper(4, 20, 0x33334a, 0x22222e));
scene.add(new THREE.HemisphereLight(0xdde4ff, 0x33333a, 1.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(1.5, 2.5, 2.0);
scene.add(keyLight);

const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.autoRotate = true;
controls.autoRotateSpeed = 1.5;

const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let current = null; // { root, vrm, skeletonHelper }
let runs = [];

function setStatus(text) { statusEl.textContent = text; }

function resize() {
  const { clientWidth, clientHeight } = stage;
  renderer.setSize(clientWidth, clientHeight);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function frameObject(root) {
  const box = new THREE.Box3().setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.1);
  const distance = (maxDim / (2 * Math.tan(camera.fov * Math.PI / 360))) * 1.35;
  camera.position.set(center.x + distance * 0.4, center.y + distance * 0.25,
                      center.z + distance);
  camera.near = distance / 100;
  camera.far = distance * 50;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function disposeCurrent() {
  if (!current) return;
  scene.remove(current.root);
  if (current.skeletonHelper) scene.remove(current.skeletonHelper);
  VRMUtils.deepDispose(current.root);
  current = null;
}

function applyWireframe(enabled) {
  if (!current) return;
  current.root.traverse((node) => {
    if (!node.isMesh) return;
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) if (material) material.wireframe = enabled;
  });
}

function applySkeleton(enabled) {
  if (!current) return;
  if (enabled && !current.skeletonHelper) {
    const helper = new THREE.SkeletonHelper(current.root);
    current.skeletonHelper = helper;
    scene.add(helper);
  } else if (!enabled && current.skeletonHelper) {
    scene.remove(current.skeletonHelper);
    current.skeletonHelper = null;
  }
}

async function loadModel(run, file) {
  setStatus(`loading ${file}…`);
  disposeCurrent();
  const url = `/api/chargen/file?run=${encodeURIComponent(run)}&f=${encodeURIComponent(file)}`;
  try {
    const gltf = await loader.loadAsync(url);
    const vrm = gltf.userData.vrm || null;
    const root = vrm ? vrm.scene : gltf.scene;
    if (vrm) VRMUtils.rotateVRM0(vrm); // no-op for VRM 1.0
    scene.add(root);
    current = { root, vrm, skeletonHelper: null };
    frameObject(root);
    applyWireframe(document.getElementById("toggle-wireframe").checked);
    applySkeleton(document.getElementById("toggle-skeleton").checked);
    setStatus(`${run} / ${file}${vrm ? " (VRM)" : ""}`);
  } catch (error) {
    console.error(error);
    setStatus(`failed to load ${file}: ${error.message || error}`);
  }
  for (const button of modelList.querySelectorAll(".model-btn")) {
    button.classList.toggle("active", button.dataset.file === file);
  }
}

function summarizeReports(entry) {
  const lines = [];
  const manifest = entry.reports?.run_manifest;
  if (manifest?.parameters) {
    const p = manifest.parameters;
    lines.push(`gen: ${p.ckpt ?? "?"} seed=${p.seed ?? "?"} octree=${p.octree_resolution ?? "?"}`);
  }
  const cleanup = entry.reports?.cleanup_report;
  if (cleanup?.output) {
    lines.push(`clean: ${cleanup.output.faces} faces / ${cleanup.output.vertices} verts`
               + ` (${cleanup.retopo_method ?? "?"})`);
    if (cleanup.input) lines.push(`  from ${cleanup.input.faces} faces`);
  }
  const rig = entry.reports?.rig_report;
  if (rig) {
    lines.push(`rig: ${rig.bones} bones, weighted ${rig.weights?.weighted_ratio ?? "?"}`
               + ` (${rig.bind_method ?? "?"})`);
  }
  if (entry.gate) {
    lines.push(`vrm gate: ${entry.gate.checks_passed}/${entry.gate.checks_total} checks`
               + (entry.gate.all_pass ? " (all pass)" : ""));
    for (const blocker of entry.gate.blockers ?? []) lines.push(`  ! ${blocker}`);
  }
  return lines.length ? lines.join("\n") : "-";
}

function showRun(runName) {
  const entry = runs.find((r) => r.run === runName);
  modelList.textContent = "";
  reportEl.textContent = entry ? summarizeReports(entry) : "-";
  if (!entry) return;
  const preferred = [...entry.models].sort((a, b) => {
    const rank = (m) => (m.name.endsWith(".vrm") ? 0 : m.name.includes("rigged") ? 1 : 2);
    return rank(a) - rank(b);
  });
  for (const model of preferred) {
    const button = document.createElement("button");
    button.className = "model-btn";
    button.dataset.file = model.name;
    button.textContent = model.name;
    const size = document.createElement("span");
    size.className = "size";
    size.textContent = `${(model.size / 1048576).toFixed(1)}M`;
    button.appendChild(size);
    button.addEventListener("click", () => loadModel(entry.run, model.name));
    modelList.appendChild(button);
  }
  if (preferred.length) loadModel(entry.run, preferred[0].name);
}

async function init() {
  resize();
  try {
    const response = await fetch("/api/chargen/runs");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    runs = (await response.json()).runs;
  } catch (error) {
    setStatus(`failed to list runs: ${error.message || error}`);
    return;
  }
  if (!runs.length) { setStatus("no chargen runs found"); return; }
  runSelect.textContent = "";
  for (const entry of runs) {
    const option = document.createElement("option");
    option.value = entry.run;
    option.textContent = entry.run;
    runSelect.appendChild(option);
  }
  runSelect.addEventListener("change", () => showRun(runSelect.value));
  showRun(runs[0].run);
}

document.getElementById("toggle-rotate").addEventListener("change", (event) => {
  controls.autoRotate = event.target.checked;
});
document.getElementById("toggle-wireframe").addEventListener("change", (event) => {
  applyWireframe(event.target.checked);
});
document.getElementById("toggle-skeleton").addEventListener("change", (event) => {
  applySkeleton(event.target.checked);
});

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const delta = clock.getDelta();
  if (current?.vrm) current.vrm.update(delta);
  controls.update();
  renderer.render(scene, camera);
});

init();
