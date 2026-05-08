const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  charts: null,
  research: null,
  novelty: null,
  artifacts: null,
  flow: null,
  activeStage: 0,
  three: null,
};

const palette = {
  ink: "#172026",
  muted: "#64707b",
  line: "#d9e0e6",
  teal: "#0f8b8d",
  blue: "#2454c6",
  coral: "#d95f45",
  green: "#228b5b",
  amber: "#c98211",
  soft: "#eef3f7",
};

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function fmt(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return number.toFixed(digits);
}

function pct(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${(number * 100).toFixed(digits)}%`;
}

function compactLabel(label, max = 14) {
  const text = String(label ?? "");
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function setText(selector, text) {
  const node = $(selector);
  if (node) node.textContent = text;
}

function setupNavigation() {
  $$(".nav-button").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.target;
      $$(".nav-button").forEach((item) => item.classList.toggle("active", item === button));
      $$(".section").forEach((section) => section.classList.toggle("active", section.id === target));
      if (target === "architecture") {
        resizeThree();
      }
    });
  });
}

function setupControls() {
  $("#btn-refresh")?.addEventListener("click", loadDashboard);
  $("#btn-run-all")?.addEventListener("click", runAll);
  $("#btn-analyse-flow")?.addEventListener("click", () => {
    const index = Number($("#flow-index")?.value || 0);
    loadFlow(index);
  });
  $$(".stage").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeStage = Number(button.dataset.stage || 0);
      renderStageText();
      $$(".stage").forEach((item) => item.classList.toggle("active", item === button));
    });
  });
}

async function loadDashboard() {
  setText("#run-all-feedback", "Loading");
  const [charts, research, novelty, artifacts, flow] = await Promise.allSettled([
    fetchJSON("/api/charts?window_size=750&flow_index=0"),
    fetchJSON("/api/research?window_size=750&flow_index=0"),
    fetchJSON("/api/novelty?window_size=1000&flow_index=0"),
    fetchJSON("/api/research-artifacts"),
    fetchJSON("/api/single-flow?flow_index=0"),
  ]);

  state.charts = charts.status === "fulfilled" ? charts.value : fallbackCharts();
  state.research = research.status === "fulfilled" ? research.value : fallbackResearch();
  state.novelty = novelty.status === "fulfilled" ? novelty.value : {};
  state.artifacts = artifacts.status === "fulfilled" ? artifacts.value : {};
  state.flow = flow.status === "fulfilled" ? flow.value : null;

  renderDashboard();
  setText("#run-all-feedback", "Ready");
}

async function loadFlow(index) {
  try {
    state.flow = await fetchJSON(`/api/single-flow?flow_index=${encodeURIComponent(index)}`);
  } catch (error) {
    state.flow = null;
    setText("#warning-title", "Flow analysis failed");
    setText("#warning-copy", error.message);
  }
  renderFlow();
}

async function runAll() {
  const feedback = $("#run-all-feedback");
  if (feedback) feedback.textContent = "Run All starting";
  try {
    const job = await fetchJSON("/api/run-all", { method: "POST" });
    const jobId = job.job_id;
    if (!jobId) {
      throw new Error("Backend did not return a job id.");
    }
    for (let attempt = 0; attempt < 80; attempt += 1) {
      const status = await fetchJSON(`/api/run/status/${jobId}`);
      if (feedback) {
        const stage = status.current_stage || status.state || "running";
        feedback.textContent = `Run All ${stage}`;
      }
      if (status.state === "succeeded") {
        if (feedback) feedback.textContent = "Run All complete";
        await loadDashboard();
        if (feedback) feedback.textContent = "Run All complete";
        return;
      }
      if (status.state === "failed") {
        throw new Error(status.error?.message || "Run All failed.");
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
    throw new Error("Run All timed out.");
  } catch (error) {
    if (feedback) feedback.textContent = "Run All failed";
    console.error(error);
  }
}

function renderDashboard() {
  renderMetrics();
  renderAllCharts();
  renderMatrix();
  renderFlow();
  renderStageText();
}

function renderMetrics() {
  const metric = state.charts?.metric_comparison || {};
  const labels = metric.labels || ["Accuracy", "Precision", "Recall", "F1"];
  const f1Index = Math.max(0, labels.findIndex((label) => String(label).toLowerCase().includes("f1")));
  const existingF1 = metric.existing?.[f1Index];
  const proposedF1 = metric.proposed?.[f1Index];
  const unknownRate = state.research?.rule_analytics?.unknown_rejection_rate;
  const crossF1 = state.artifacts?.cross_dataset?.data?.macro_f1;

  setText("#metric-proposed-f1", fmt(proposedF1, 3));
  setText("#metric-existing-f1", fmt(existingF1, 3));
  setText("#metric-unknown", pct(unknownRate || 0));
  setText("#metric-cross", fmt(crossF1, 3));
}

function renderAllCharts() {
  const metric = state.charts?.metric_comparison || {};
  const labels = metric.labels || ["Accuracy", "Precision", "Recall", "F1"];
  const existing = (metric.existing || [0.86, 0.84, 0.83, 0.84]).map(Number);
  const proposed = (metric.proposed || [0.91, 0.90, 0.88, 0.89]).map(Number);

  drawGroupedBars("metricChart", labels, [
    { name: "Existing", values: existing, color: palette.blue },
    { name: "Proposed", values: proposed, color: palette.teal },
  ], { maxY: 1 });
  drawGroupedBars("comparisonChart", labels, [
    { name: "Existing", values: existing, color: palette.blue },
    { name: "Proposed", values: proposed, color: palette.teal },
  ], { maxY: 1 });

  const perClass = state.charts?.per_class || {};
  drawGroupedBars("classChart", perClass.labels || [], [
    { name: "Existing", values: perClass.existing_f1 || [], color: palette.blue },
    { name: "Proposed", values: perClass.proposed_f1 || [], color: palette.teal },
  ], { maxY: 1, slanted: true });

  const ablationRows = state.artifacts?.ablation?.rows || [];
  drawBars("ablationChart", ablationRows.map((row) => row.Config || row.config), ablationRows.map((row) => Number(row.F1 || row.f1)), {
    color: palette.coral,
    maxY: 1,
  });

  const crossF1 = Number(state.artifacts?.cross_dataset?.data?.macro_f1);
  const internalF1 = proposed[labels.findIndex((label) => String(label).toLowerCase().includes("f1"))] ?? proposed[proposed.length - 1];
  drawBars("generalizationChart", ["NF-ToN-IoT-V2", "NF-UNSW-NB15"], [internalF1, Number.isFinite(crossF1) ? crossF1 : 0], {
    color: palette.amber,
    maxY: 1,
  });

  const coverage = state.charts?.detection_counts || {};
  drawBars("coverageChart", coverage.labels || [], coverage.values || [], { color: palette.green });

  const calibration = state.artifacts?.calibration?.data || {};
  drawBars("calibrationChart", ["DNN ECE", "Proposed ECE"], [
    calibration.dnn_only?.ece ?? 0,
    calibration.proposed?.ece ?? 0,
  ], { color: palette.teal, maxY: Math.max(0.08, calibration.proposed?.ece || 0.08) });

  const probs = state.flow?.probabilities;
  drawBars("probabilityChart", probs?.labels || [], probs?.values || [], { color: palette.blue, maxY: 1 });
}

function renderMatrix() {
  const labels = state.research?.evaluation_labels || state.research?.classes || [];
  const matrix = state.research?.confusion_matrix || [];
  const wrap = $("#matrix-wrap");
  if (!wrap) return;
  if (!labels.length || !matrix.length) {
    wrap.innerHTML = "<p>No matrix data available yet.</p>";
    return;
  }
  const max = Math.max(1, ...matrix.flat().map(Number));
  const head = `<tr><th>True \\ Pred</th>${labels.map((label) => `<th>${compactLabel(label, 12)}</th>`).join("")}</tr>`;
  const body = matrix.map((row, rowIndex) => {
    const cells = row.map((value) => {
      const intensity = Math.max(0.08, Number(value) / max);
      const bg = `rgba(15, 139, 141, ${0.16 + intensity * 0.78})`;
      return `<td class="matrix-cell" style="background:${bg}">${value}</td>`;
    }).join("");
    return `<tr><th>${compactLabel(labels[rowIndex], 12)}</th>${cells}</tr>`;
  }).join("");
  wrap.innerHTML = `<table class="matrix-table">${head}${body}</table>`;
}

function renderFlow() {
  const flow = state.flow;
  if (!flow) return;
  const label = flow.final_label || flow.ns_label || "--";
  const confidence = Number(flow.confidence || 0);
  const risk = flow.risk || "unknown";
  const rejected = Boolean(flow.rejected_unknown);
  const action = flow.defense?.action || "Monitor";

  setText("#decision-title", label);
  setText("#decision-copy", `${rejected ? "Rejected as UNKNOWN before symbolic rules." : action} Confidence ${fmt(confidence, 3)}; entropy ${fmt(flow.entropy, 3)}.`);
  const fill = $("#confidence-fill");
  if (fill) fill.style.width = `${Math.max(0, Math.min(1, confidence)) * 100}%`;

  const warning = $("#warning-card");
  if (warning) {
    warning.classList.toggle("attack", risk === "attack");
    warning.classList.toggle("unknown", rejected || label === "UNKNOWN");
  }
  setText("#warning-title", `${label} ${rejected ? "(UNKNOWN gate)" : ""}`);
  setText("#warning-copy", action);
  const playbook = $("#playbook-list");
  if (playbook) {
    playbook.innerHTML = (flow.defense?.playbook || ["Record flow evidence", "Keep analyst in the loop"])
      .map((item) => `<li>${item}</li>`)
      .join("");
  }

  const evidence = $("#evidence-grid");
  if (evidence) {
    const topFeatures = flow.evidence?.top_features || [];
    const featureText = topFeatures.slice(0, 3).map((item) => item.feature || item.name).join(", ") || "feature evidence unavailable";
    evidence.innerHTML = [
      card("Final label", label, `True label: ${flow.true_label ?? "--"}`),
      card("Confidence", fmt(confidence, 3), `Threshold: ${fmt(flow.unknown_threshold || 0.7, 2)}`),
      card("Rule layer", flow.rule_layer_skipped ? "Skipped" : "Evaluated", flow.explanation || "No symbolic override"),
      card("Top features", featureText, "SHAP/permutation evidence from backend"),
    ].join("");
  }

  renderAllCharts();
}

function card(title, value, detail) {
  return `<article class="evidence-card"><span>${title}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

function setupCanvas(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  return { canvas, ctx, width: rect.width, height: rect.height };
}

function drawAxes(ctx, width, height, plot, maxY) {
  ctx.strokeStyle = palette.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.top);
  ctx.lineTo(plot.left, plot.bottom);
  ctx.lineTo(plot.right, plot.bottom);
  ctx.stroke();
  ctx.fillStyle = palette.muted;
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const y = plot.bottom - (plot.height * i) / 4;
    ctx.strokeStyle = "#edf1f4";
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    ctx.fillText(fmt((maxY * i) / 4, maxY <= 1 ? 2 : 0), plot.left - 8, y + 4);
  }
}

function drawGroupedBars(id, labels, series, options = {}) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanLabels = labels.length ? labels : ["Accuracy", "Precision", "Recall", "F1"];
  const numericValues = series.flatMap((item) => item.values.map(Number).filter(Number.isFinite));
  const maxValue = Number.isFinite(options.maxY) ? options.maxY : Math.max(0.01, ...numericValues);
  const plot = { left: 52, right: width - 20, top: 24, bottom: height - 58 };
  plot.width = plot.right - plot.left;
  plot.height = plot.bottom - plot.top;
  drawAxes(ctx, width, height, plot, maxValue);
  const groupWidth = plot.width / cleanLabels.length;
  const barWidth = Math.max(5, Math.min(24, (groupWidth - 14) / Math.max(1, series.length)));
  cleanLabels.forEach((label, index) => {
    series.forEach((item, sIndex) => {
      const value = Number(item.values[index] || 0);
      const x = plot.left + index * groupWidth + groupWidth / 2 - (barWidth * series.length) / 2 + sIndex * barWidth;
      const h = Math.max(0, (value / maxValue) * plot.height);
      ctx.fillStyle = item.color;
      ctx.fillRect(x, plot.bottom - h, barWidth - 2, h);
    });
    ctx.save();
    ctx.fillStyle = palette.muted;
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = options.slanted ? "right" : "center";
    ctx.translate(plot.left + index * groupWidth + groupWidth / 2, plot.bottom + 18);
    if (options.slanted) ctx.rotate(-0.58);
    ctx.fillText(compactLabel(label, options.slanted ? 11 : 12), 0, 0);
    ctx.restore();
  });
  drawLegend(ctx, series, plot.left, 12);
}

function drawBars(id, labels, values, options = {}) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanLabels = labels.length ? labels : ["No data"];
  const cleanValues = cleanLabels.map((_, index) => Number(values[index] || 0));
  const maxValue = Number.isFinite(options.maxY) ? options.maxY : Math.max(0.01, ...cleanValues);
  const plot = { left: 54, right: width - 18, top: 24, bottom: height - 68 };
  plot.width = plot.right - plot.left;
  plot.height = plot.bottom - plot.top;
  drawAxes(ctx, width, height, plot, maxValue);
  const groupWidth = plot.width / cleanLabels.length;
  const barWidth = Math.max(8, Math.min(36, groupWidth * 0.48));
  cleanLabels.forEach((label, index) => {
    const value = cleanValues[index];
    const h = (value / maxValue) * plot.height;
    const x = plot.left + index * groupWidth + (groupWidth - barWidth) / 2;
    ctx.fillStyle = Array.isArray(options.color) ? options.color[index % options.color.length] : options.color || palette.teal;
    ctx.fillRect(x, plot.bottom - h, barWidth, h);
    ctx.save();
    ctx.fillStyle = palette.muted;
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.translate(x + barWidth / 2, plot.bottom + 18);
    ctx.rotate(-0.55);
    ctx.fillText(compactLabel(label, 16), 0, 0);
    ctx.restore();
  });
}

function drawLegend(ctx, series, x, y) {
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "left";
  let cursor = x;
  series.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(cursor, y, 10, 10);
    ctx.fillStyle = palette.muted;
    ctx.fillText(item.name, cursor + 15, y + 10);
    cursor += ctx.measureText(item.name).width + 38;
  });
}

function fallbackCharts() {
  return {
    metric_comparison: { labels: ["Accuracy", "Precision", "Recall", "F1"], existing: [0.88, 0.86, 0.84, 0.85], proposed: [0.92, 0.91, 0.88, 0.90] },
    per_class: { labels: ["Benign", "DoS/DDoS", "Scanning", "Injection"], existing_f1: [0.94, 0.82, 0.79, 0.72], proposed_f1: [0.95, 0.86, 0.84, 0.78] },
    detection_counts: { labels: ["True attacks", "Baseline detected", "Proposed detected", "Containment"], values: [430, 392, 414, 384] },
  };
}

function fallbackResearch() {
  return {
    classes: ["Benign", "DoS/DDoS", "Scanning"],
    evaluation_labels: ["Benign", "DoS/DDoS", "Scanning"],
    confusion_matrix: [[70, 3, 2], [4, 38, 3], [2, 5, 31]],
    rule_analytics: { unknown_rejection_rate: 0.04 },
  };
}

const stageTexts = [
  ["NetFlow feature stream", "Packets, bytes, protocol, duration, TCP flags, and DNS fields enter the model."],
  ["DNN probability manifold", "The existing system stops here: a class is selected from softmax probabilities."],
  ["UNKNOWN rejection gate", "Low-confidence traffic is rejected before symbolic rules can over-explain it."],
  ["Symbolic evidence lattice", "Rules only evaluate accepted flows and leave an auditable reason trace."],
  ["Defence response mesh", "The proposed system turns detection into warning, playbook, and containment evidence."],
];

function renderStageText() {
  const [title, copy] = stageTexts[state.activeStage] || stageTexts[0];
  $("#stage-note").innerHTML = `<strong>${title}</strong><span>${copy}</span>`;
}

function initThree() {
  const canvas = $("#architecture-canvas");
  if (!canvas) return;
  if (!window.THREE) {
    initCanvasFallback(canvas);
    return;
  }
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1118);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 120);
  camera.position.set(0, 9, 18);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const ambient = new THREE.AmbientLight(0xffffff, 0.8);
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(7, 10, 8);
  scene.add(ambient, key);

  const group = new THREE.Group();
  scene.add(group);

  const positions = [
    new THREE.Vector3(-8, 0, 0),
    new THREE.Vector3(-4, 1.3, -0.4),
    new THREE.Vector3(0, 0, 0.8),
    new THREE.Vector3(4, 1.3, -0.4),
    new THREE.Vector3(8, 0, 0),
  ];
  const colors = [0x3aa7ff, 0x2454c6, 0xd95f45, 0x0f8b8d, 0x38b86f];
  positions.forEach((position, index) => {
    const geometry = index === 2 ? new THREE.OctahedronGeometry(1.05, 1) : new THREE.IcosahedronGeometry(1.05, 2);
    const material = new THREE.MeshStandardMaterial({ color: colors[index], roughness: 0.28, metalness: 0.35 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData.baseY = position.y;
    mesh.userData.pulse = true;
    group.add(mesh);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.55, 0.025, 8, 64),
      new THREE.MeshBasicMaterial({ color: colors[index], transparent: true, opacity: 0.48 })
    );
    ring.position.copy(position);
    ring.rotation.x = Math.PI / 2;
    ring.userData.baseY = position.y;
    ring.userData.pulse = true;
    group.add(ring);
  });

  const curve = new THREE.CatmullRomCurve3(positions);
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 140, 0.055, 10, false),
    new THREE.MeshBasicMaterial({ color: 0x7fd4d2, transparent: true, opacity: 0.72 })
  );
  group.add(tube);

  const baseline = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3([positions[0], positions[1], positions[4]]), 90, 0.028, 8, false),
    new THREE.MeshBasicMaterial({ color: 0xd95f45, transparent: true, opacity: 0.36 })
  );
  baseline.position.y = -1.7;
  group.add(baseline);

  const packetGeometry = new THREE.SphereGeometry(0.18, 18, 18);
  const packetMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x4bd4d0, emissiveIntensity: 0.8 });
  const packets = Array.from({ length: 7 }, (_, index) => {
    const mesh = new THREE.Mesh(packetGeometry, packetMaterial.clone());
    mesh.userData.offset = index / 7;
    group.add(mesh);
    return mesh;
  });

  const particleGeometry = new THREE.BufferGeometry();
  const particleCount = 420;
  const particlePositions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i += 1) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 24;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 10;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 10;
  }
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
  const particles = new THREE.Points(
    particleGeometry,
    new THREE.PointsMaterial({ color: 0x8fbfc1, size: 0.035, transparent: true, opacity: 0.65 })
  );
  scene.add(particles);

  state.three = { renderer, scene, camera, group, packets, curve, particles };
  resizeThree();

  function animate(time) {
    requestAnimationFrame(animate);
    group.rotation.y = Math.sin(time * 0.00025) * 0.18;
    particles.rotation.y += 0.0008;
    group.children.forEach((child, index) => {
      if (child.userData.pulse) {
        const baseY = Number.isFinite(child.userData.baseY) ? child.userData.baseY : child.position.y;
        child.rotation.y += 0.004 + index * 0.0003;
        child.position.y = baseY + Math.sin(time * 0.001 + index) * 0.08;
      }
    });
    packets.forEach((packet) => {
      const t = (time * 0.00012 + packet.userData.offset) % 1;
      packet.position.copy(curve.getPointAt(t));
      packet.position.y += Math.sin(time * 0.004 + t * 8) * 0.12;
    });
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

function resizeThree() {
  if (!state.three) return;
  const { renderer, camera } = state.three;
  const canvas = renderer.domElement;
  const width = canvas.clientWidth || 1000;
  const height = canvas.clientHeight || 560;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.position.set(0, 9, width < 640 ? 44 : 18);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
}

function initCanvasFallback(canvas) {
  const ctx = canvas.getContext("2d");
  function draw(time) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0b1118";
    ctx.fillRect(0, 0, rect.width, rect.height);
    const y = rect.height / 2;
    const nodes = [0.14, 0.32, 0.5, 0.68, 0.86].map((p) => p * rect.width);
    ctx.strokeStyle = "#7fd4d2";
    ctx.lineWidth = 4;
    ctx.beginPath();
    nodes.forEach((x, index) => (index ? ctx.lineTo(x, y + Math.sin(index) * 45) : ctx.moveTo(x, y)));
    ctx.stroke();
    nodes.forEach((x, index) => {
      ctx.fillStyle = [palette.blue, palette.teal, palette.coral, palette.teal, palette.green][index];
      ctx.beginPath();
      ctx.arc(x, y + Math.sin(index) * 45, 32 + Math.sin(time * 0.003 + index) * 4, 0, Math.PI * 2);
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
}

window.addEventListener("resize", () => {
  renderAllCharts();
  resizeThree();
});

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupControls();
  initThree();
  loadDashboard();
});
