const state = {
  overview: null,
  research: null,
  chartData: null,
  backend: null,
  flow: null,
  incident: null,
  charts: {},
  maxIndex: 0,
  architectureTimer: null,
};

const colors = {
  blue: "#4ba3ff",
  cyan: "#19d3c5",
  green: "#4ade80",
  amber: "#f4b740",
  red: "#ff5b6e",
  violet: "#9b87ff",
  muted: "#8fa2b3",
};

if (window.Chart) {
  Chart.defaults.color = colors.muted;
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
}

function $(selector) {
  return document.querySelector(selector);
}

function $$(selector) {
  return Array.from(document.querySelectorAll(selector));
}

function pct(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

async function getJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

function setView(name) {
  $$(".nav-item").forEach(btn => btn.classList.toggle("active", btn.dataset.view === name));
  $$(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
  history.replaceState(null, "", name === "overview" ? "/" : `#${name}`);
}

function setupNavigation() {
  $$(".nav-item").forEach(btn => btn.addEventListener("click", () => setView(btn.dataset.view)));
  $$("[data-view-jump]").forEach(btn => btn.addEventListener("click", () => setView(btn.dataset.viewJump)));
  const initial = location.hash.replace("#", "");
  if (initial && $(`#view-${initial}`)) setView(initial);
}

function destroyChart(key) {
  if (state.charts[key]) state.charts[key].destroy();
}

function makeChart(key, canvasId, config) {
  if (!window.Chart) {
    const canvas = document.getElementById(canvasId);
    if (canvas) {
      const box = canvas.closest(".chart-box");
      if (box) box.innerHTML = `<div class="chart-fallback">Chart library unavailable. Backend data and CSV exports still work.</div>`;
    }
    return;
  }
  destroyChart(key);
  state.charts[key] = new Chart(document.getElementById(canvasId), config);
}

function chartScales() {
  return {
    x: { grid: { color: "rgba(255,255,255,.05)" } },
    y: { grid: { color: "rgba(255,255,255,.05)" }, beginAtZero: true },
  };
}

function renderOverview() {
  const o = state.overview;
  const r = state.research;
  const existingAccuracy = r.metrics.existing[0] || 0;
  const proposedAccuracy = r.metrics.proposed[0] || 0;
  const lift = proposedAccuracy - existingAccuracy;

  $("#headline-score").textContent = `${pct(proposedAccuracy)} backend-computed accuracy with neuro-symbolic explanation`;
  $("#kpi-lift").textContent = `+${(lift * 100).toFixed(2)} pts`;
  $("#kpi-f1").textContent = pct(r.metrics.proposed[3]);
  $("#kpi-classes").textContent = o.num_classes;
  $("#kpi-samples").textContent = Number(o.total_samples || 0).toLocaleString();
  $("#flow-index").max = o.max_index;
  state.maxIndex = o.max_index;

  makeChart("comparisonChart", "chart-comparison", {
    type: "bar",
    data: {
      labels: r.metrics.labels,
      datasets: [
        {
          label: "Existing system",
          data: r.metrics.existing,
          backgroundColor: "rgba(244,183,64,.72)",
          borderColor: colors.amber,
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: "Proposed neuro-symbolic",
          data: r.metrics.proposed,
          backgroundColor: "rgba(25,211,197,.72)",
          borderColor: colors.cyan,
          borderWidth: 1,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { ...chartScales(), y: { min: 0.75, max: 1, grid: { color: "rgba(255,255,255,.05)" } } },
      plugins: { tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${pct(ctx.raw, 2)}` } } },
    },
  });

  makeChart("distributionChart", "chart-distribution", {
    type: "doughnut",
    data: {
      labels: r.class_distribution.labels,
      datasets: [{
        data: r.class_distribution.values,
        backgroundColor: [colors.blue, colors.green, colors.red, colors.violet, colors.amber, colors.cyan, "#ff7ac8"],
        borderColor: "#101720",
        borderWidth: 2,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: "58%" },
  });
  renderBackendStatus();
}

function renderAnalysis() {
  const r = state.research;
  const c = state.chartData;
  $("#cache-status").textContent = `${r.limit.toLocaleString()} flows cached`;
  $("#sample-window-value").textContent = r.limit;
  $("#sample-window").value = r.limit;

  makeChart("rulesChart", "chart-rules", {
    type: "bar",
    data: {
      labels: r.rule_hits.labels.length ? r.rule_hits.labels : ["NONE"],
      datasets: [{
        label: "Rule hits",
        data: r.rule_hits.values.length ? r.rule_hits.values : [0],
        backgroundColor: "rgba(155,135,255,.72)",
        borderColor: colors.violet,
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: chartScales(), plugins: { legend: { display: false } } },
  });

  makeChart("analysisComparisonChart", "chart-analysis-comparison", {
    type: "line",
    data: {
      labels: c.improvement_curve.labels,
      datasets: [
        {
          label: "Existing system",
          data: c.improvement_curve.existing_accuracy,
          borderColor: colors.amber,
          backgroundColor: "rgba(244,183,64,.16)",
          tension: 0.35,
          fill: true,
        },
        {
          label: "Proposed neuro-symbolic",
          data: c.improvement_curve.proposed_accuracy,
          borderColor: colors.cyan,
          backgroundColor: "rgba(25,211,197,.14)",
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Evaluation window size" }, grid: { color: "rgba(255,255,255,.05)" } },
        y: { min: 0.82, max: 0.94, title: { display: true, text: "Accuracy" }, grid: { color: "rgba(255,255,255,.05)" } },
      },
      plugins: { tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${pct(ctx.raw, 2)}` } } },
    },
  });

  makeChart("perClassChart", "chart-per-class", {
    type: "radar",
    data: {
      labels: c.per_class.labels,
      datasets: [
        { label: "Backend baseline MLP F1", data: c.per_class.existing_f1, backgroundColor: "rgba(244,183,64,.14)", borderColor: colors.amber, borderWidth: 2 },
        { label: "Neuro-symbolic F1", data: c.per_class.paper_proposed_f1, backgroundColor: "rgba(25,211,197,.18)", borderColor: colors.cyan, borderWidth: 2 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { r: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.08)" }, angleLines: { color: "rgba(255,255,255,.08)" }, pointLabels: { color: colors.muted } } },
    },
  });

  makeChart("confidenceChart", "chart-confidence", {
    type: "line",
    data: {
      labels: c.confidence_histogram.labels,
      datasets: [{ label: "Flows", data: c.confidence_histogram.values, backgroundColor: "rgba(75,163,255,.16)", borderColor: colors.blue, borderWidth: 2, tension: 0.35, fill: true }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: chartScales(), plugins: { legend: { display: false } } },
  });

  makeChart("detectionChart", "chart-detection", {
    type: "doughnut",
    data: {
      labels: c.detection_counts.labels,
      datasets: [{ label: "Flows", data: c.detection_counts.values, backgroundColor: ["rgba(255,91,110,.72)", "rgba(244,183,64,.72)", "rgba(25,211,197,.72)", "rgba(74,222,128,.72)"], borderColor: "#101720", borderWidth: 2 }],
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: "54%" },
  });

  makeChart("errorRateChart", "chart-error-rate", {
    type: "bar",
    data: {
      labels: c.class_error_rate.labels,
      datasets: [{ label: "Error rate", data: c.class_error_rate.values, backgroundColor: "rgba(255,91,110,.62)", borderColor: colors.red, borderWidth: 1, borderRadius: 5 }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { ...chartScales(), y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.05)" } } }, plugins: { tooltip: { callbacks: { label: ctx => pct(ctx.raw, 2) } } } },
  });

  makeChart("rocChart", "chart-roc", {
    type: "line",
    data: {
      datasets: [
        { label: `ROC AUC ${c.roc_curve.auc ?? "n/a"}`, data: c.roc_curve.points, borderColor: colors.green, backgroundColor: "rgba(74,222,128,.12)", parsing: false, pointRadius: 0, tension: 0.25, fill: true },
        { label: "Random baseline", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "rgba(143,162,179,.55)", borderDash: [4, 4], pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" }, grid: { color: "rgba(255,255,255,.05)" } },
        y: { min: 0, max: 1, title: { display: true, text: "True positive rate" }, grid: { color: "rgba(255,255,255,.05)" } },
      },
    },
  });

  renderMatrix(r.classes, r.confusion_matrix);
  renderAuditTable(r.rows);
}

function renderBackendStatus() {
  if (!state.backend) return;
  const b = state.backend;
  const items = [
    ["Backend", b.backend],
    ["Model loaded", b.model_loaded ? "Yes" : "No"],
    ["Rows", Number(b.test_rows || 0).toLocaleString()],
    ["Features", b.feature_count],
    ["Classes", (b.classes || []).join(", ")],
    ["Analysis cache", (b.cached_analysis_windows || []).join(", ") || "cold"],
    ["Chart cache", (b.cached_chart_windows || []).join(", ") || "cold"],
    ["Incidents", b.incident_count],
    ["Model path", b.model_path],
    ["Data path", b.test_path],
  ];
  $("#backend-grid").innerHTML = items.map(([key, value]) => `
    <div class="backend-item"><span>${key}</span><strong>${value}</strong></div>
  `).join("");
}

function renderMatrix(labels, matrix) {
  const max = Math.max(...matrix.flat(), 1);
  const wrap = $("#matrix-wrap");
  wrap.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "matrix-grid";
  grid.style.gridTemplateColumns = `120px repeat(${labels.length}, minmax(58px, 1fr))`;
  grid.appendChild(matrixCell("", true));
  labels.forEach(label => grid.appendChild(matrixCell(label, true)));
  labels.forEach((label, rowIndex) => {
    grid.appendChild(matrixCell(label, true));
    matrix[rowIndex].forEach(value => {
      const cell = matrixCell(value.toLocaleString(), false);
      const alpha = 0.06 + (value / max) * 0.72;
      cell.style.background = `rgba(25, 211, 197, ${alpha})`;
      grid.appendChild(cell);
    });
  });
  wrap.appendChild(grid);
}

function matrixCell(text, head) {
  const div = document.createElement("div");
  div.className = `matrix-cell${head ? " head" : ""}`;
  div.textContent = text;
  return div;
}

function renderAuditTable(rows) {
  $("#audit-body").innerHTML = rows.map(row => `
    <tr>
      <td>${row.idx}</td>
      <td>${row.true}</td>
      <td>${row.baseline}</td>
      <td>${row.proposed}</td>
      <td class="risk-${row.risk}">${row.risk.toUpperCase()}</td>
    </tr>
  `).join("");
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

async function analyseFlow(index = $("#flow-index").value) {
  const result = await postJson("/api/defense/analyse", { idx: index });
  const flow = result.flow;
  state.flow = flow;
  state.incident = result.incident;
  $("#flow-index").value = flow.index;
  renderDefense(flow, result.incident);
  return flow;
}

function renderDefense(flow, incident) {
  const isAttack = flow.risk === "attack";
  const confidence = Number(flow.confidence || 0);
  const panel = $("#warning-panel");
  panel.classList.toggle("attack", isAttack);
  panel.classList.toggle("benign", !isAttack);
  $("#btn-contain-flow").textContent = isAttack ? "Simulate Containment" : "Allow and Monitor";

  $("#decision-label").textContent = isAttack
    ? `${flow.ns_label} detected at ${pct(confidence)} confidence`
    : `Benign flow confirmed at ${pct(confidence)} confidence`;
  $("#decision-action").textContent = flow.defense.action;
  $("#playbook-list").innerHTML = flow.defense.playbook.map(item => `<li>${item}</li>`).join("");
  renderIncident(incident);

  const threatCard = $("#threat-card");
  threatCard.classList.toggle("danger", isAttack);
  threatCard.classList.toggle("safe", !isAttack);
  $("#threat-title").textContent = isAttack ? `Warning: ${flow.ns_label}` : "No active attack";
  $("#threat-copy").textContent = isAttack ? flow.defense.action : "Flow is allowed while telemetry remains under observation.";
  $("#threat-meter-fill").style.width = `${Math.max(8, confidence * 100)}%`;
  $("#threat-meter-fill").style.background = isAttack ? colors.red : colors.green;

  makeChart("probabilityChart", "chart-probabilities", {
    type: "bar",
    data: {
      labels: flow.probabilities.labels,
      datasets: [{
        label: "Probability",
        data: flow.probabilities.values,
        backgroundColor: flow.probabilities.labels.map(label => label === flow.ns_label ? "rgba(255,91,110,.78)" : "rgba(75,163,255,.55)"),
        borderColor: flow.probabilities.labels.map(label => label === flow.ns_label ? colors.red : colors.blue),
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: { x: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.05)" } }, y: { grid: { display: false } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => pct(ctx.raw, 2) } } },
    },
  });

  const rules = flow.fired_rules.map(rule => `${rule.rule_id}: ${rule.reason}`).join(" | ");
  const topFeatures = Object.entries(flow.features).slice(0, 14);
  $("#evidence-grid").innerHTML = [
    ["Flow index", flow.index],
    ["True label", flow.true_label],
    ["Existing prediction", flow.base_pred],
    ["Proposed prediction", flow.ns_label],
    ["Robust model", flow.robust_pred || "Unavailable"],
    ["Symbolic trace", rules],
    ...topFeatures,
  ].map(([key, value]) => `<div class="evidence-item"><span>${key}</span><strong>${value}</strong></div>`).join("");
}

function renderIncident(incident) {
  if (!incident) return;
  $("#incident-strip").textContent = `${incident.incident_id} | ${incident.status} | ${incident.severity}`;
  $("#defense-timeline").innerHTML = [
    ...incident.timeline.map(item => `<div class="timeline-item"><strong>${item.time}</strong><span>${item.event}</span></div>`),
    ...incident.controls.map(control => `<div class="timeline-item control-${control.state}"><strong>${control.state}</strong><span>${control.name}</span></div>`),
  ].join("");
}

const stageCopy = [
  {
    title: "Flow ingestion",
    copy: "Telemetry is normalised into NF-ToN-IoT-V2 NetFlow features before neural inference.",
    steps: ["Capture flow tuple and traffic rates", "Standardise feature scale", "Preserve feature vector for explanation"],
  },
  {
    title: "Neural inference",
    copy: "The trained MLP estimates class probabilities across benign and attack families.",
    steps: ["Run baseline classifier", "Rank candidate attack classes", "Expose confidence distribution to the dashboard"],
  },
  {
    title: "Symbolic reasoning",
    copy: "Domain rules correct or explain neural decisions using packet-rate, byte-rate, duration, and anomaly context.",
    steps: ["Check high-rate DDoS bursts", "Check slow sustained attacks", "Attach fired-rule trace to the prediction"],
  },
  {
    title: "Defence response",
    copy: "The final label is converted into practical containment guidance for analyst review.",
    steps: ["Warn user when attack is detected", "Recommend isolation, rate limiting, or blocking", "Export evidence for incident reporting"],
  },
];

function setArchitectureStage(stage) {
  $$(".arch-step").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.stage) === stage));
  $$(".node").forEach((node, index) => node.classList.toggle("active", index === stage));
  $("#stage-title").textContent = stageCopy[stage].title;
  $("#stage-copy").textContent = stageCopy[stage].copy;
  $("#stage-steps").innerHTML = stageCopy[stage].steps.map(step => `<li>${step}</li>`).join("");
  const scene = $("#scene");
  if (scene) scene.dataset.stage = stage;
}

function startArchitectureLoop() {
  if (state.architectureTimer) clearInterval(state.architectureTimer);
  let stage = 0;
  state.architectureTimer = setInterval(() => {
    if (!$("#view-architecture").classList.contains("active")) return;
    stage = (stage + 1) % stageCopy.length;
    setArchitectureStage(stage);
  }, 2600);
}

function download(filename, content, type = "text/plain") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function rowsToCsv(rows) {
  return rows.map(row => row.map(value => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",")).join("\n");
}

function setupExports() {
  $("#btn-export-json").addEventListener("click", () => {
    download("neuro_symbolic_dashboard_export.json", JSON.stringify({ overview: state.overview, research: state.research, flow: state.flow }, null, 2), "application/json");
  });
  $("#btn-export-flow").addEventListener("click", () => {
    if (!state.flow) return;
    const rows = [["field", "value"], ...Object.entries(state.flow.features), ["true_label", state.flow.true_label], ["proposed_label", state.flow.ns_label], ["action", state.flow.defense.action]];
    download("flow_evidence.csv", rowsToCsv(rows), "text/csv");
  });
  $$("[data-export-chart]").forEach(btn => btn.addEventListener("click", () => {
    const chart = state.charts[btn.dataset.exportChart];
    if (!chart) return;
    const link = document.createElement("a");
    link.download = `${btn.dataset.exportChart}.png`;
    link.href = chart.toBase64Image("image/png", 1);
    link.click();
  }));
  $("#btn-export-matrix-csv").addEventListener("click", () => exportMatrixCsv());
  $$("[data-export-table]").forEach(btn => btn.addEventListener("click", () => {
    if (!state.research) return;
    if (btn.dataset.exportTable === "matrix") {
      exportMatrixCsv();
    } else {
      const rows = [["idx", "true", "existing", "proposed", "risk"], ...state.research.rows.map(row => [row.idx, row.true, row.baseline, row.proposed, row.risk])];
      download("audit_table.csv", rowsToCsv(rows), "text/csv");
    }
  }));
}

function exportMatrixCsv() {
  if (!state.research) return;
  const rows = [["true/pred", ...state.research.classes], ...state.research.classes.map((label, i) => [label, ...state.research.confusion_matrix[i]])];
  download("confusion_matrix.csv", rowsToCsv(rows), "text/csv");
}

function setupControls() {
  $("#btn-refresh").addEventListener("click", loadDashboard);
  $("#btn-backend-refresh").addEventListener("click", async () => {
    state.backend = await getJson("/api/backend/status");
    renderBackendStatus();
  });
  $("#sample-window").addEventListener("input", event => $("#sample-window-value").textContent = event.target.value);
  $("#sample-window").addEventListener("change", async event => {
    [state.research, state.chartData, state.backend] = await Promise.all([
      getJson(`/api/research?limit=${event.target.value}`),
      getJson(`/api/charts?limit=${event.target.value}`),
      getJson("/api/backend/status"),
    ]);
    renderOverview();
    renderAnalysis();
  });
  $("#btn-analyse-flow").addEventListener("click", () => analyseFlow());
  $("#btn-random-flow").addEventListener("click", () => analyseFlow(Math.floor(Math.random() * (state.maxIndex + 1))));
  $("#btn-contain-flow").addEventListener("click", async () => {
    if (!state.incident) return;
    const result = await postJson("/api/defense/contain", { incident_id: state.incident.incident_id });
    state.incident = result.incident;
    $("#decision-action").textContent = result.message;
    renderIncident(result.incident);
  });
  $("#btn-defense-status").addEventListener("click", async () => {
    const status = await getJson("/api/defense/status");
    $("#cache-status").textContent = `${status.total_incidents} backend incidents tracked`;
  });
  $$(".arch-step").forEach(btn => btn.addEventListener("click", () => setArchitectureStage(Number(btn.dataset.stage))));
}

async function loadDashboard() {
  $("#cache-status").textContent = "Loading research cache";
  const [overview, research, chartData, backend] = await Promise.all([
    getJson("/api/overview"),
    getJson(`/api/research?limit=${$("#sample-window")?.value || 750}`),
    getJson(`/api/charts?limit=${$("#sample-window")?.value || 750}`),
    getJson("/api/backend/status"),
  ]);
  state.overview = overview;
  state.research = research;
  state.chartData = chartData;
  state.backend = backend;
  renderOverview();
  renderAnalysis();
  await analyseFlow(0);
  setArchitectureStage(0);
  startArchitectureLoop();
  $("#cache-status").textContent = `${research.limit.toLocaleString()} flows cached`;
}

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupControls();
  setupExports();
  loadDashboard().catch(error => {
    console.error(error);
    $("#cache-status").textContent = "Dashboard data failed to load";
  });
});
