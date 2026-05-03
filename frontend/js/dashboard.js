const state = {
  overview: null,
  research: null,
  chartData: null,
  novelty: null,
  backend: null,
  flow: null,
  incident: null,
  charts: {},
  maxIndex: 0,
  architectureTimer: null,
  architectureScene: null,
  windowRequestId: 0,
  lastRunDebug: null,
};

const DEBOUNCE_MS = 350;

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

function setStatus(message) {
  const el = $("#cache-status");
  if (el) el.textContent = message;
}

function showActionError(error) {
  console.error(error);
  setStatus(`Action failed: ${error.message || error}`);
}

function bindAsyncClick(selector, statusText, handler) {
  const el = $(selector);
  if (!el) return;
  el.addEventListener("click", async event => {
    try {
      el.disabled = true;
      setStatus(statusText);
      await handler(event);
    } catch (error) {
      showActionError(error);
    } finally {
      el.disabled = false;
    }
  });
}

function pct(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function signedPct(value, digits = 2) {
  const number = Number(value || 0) * 100;
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)} pts`;
}

async function getJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

function currentLimit() {
  return Number($("#sample-window")?.value || 750);
}

function currentAlpha() {
  const value = Number($("#novelty-alpha")?.value || 0.65);
  if (!Number.isFinite(value)) return 0.65;
  return Math.min(0.95, Math.max(0.05, value));
}

function currentBeta() {
  const value = Number($("#fusion-beta")?.value);
  if (Number.isFinite(value)) return Math.min(0.95, Math.max(0.0, value));
  return Number((1 - currentAlpha()).toFixed(2));
}

function currentFusionMode() {
  return $("#fusion-mode")?.checked ? "soft" : "hard";
}

function currentSeed() {
  const value = Number($("#seed-selector")?.value || 60);
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 60;
}

function currentFlowIndex() {
  const value = Number($("#flow-index")?.value || 0);
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
}

function currentParams(overrides = {}) {
  return {
    window_size: currentLimit(),
    flow_index: currentFlowIndex(),
    alpha: currentAlpha(),
    beta: currentBeta(),
    fusion_mode: currentFusionMode(),
    seed: currentSeed(),
    ...overrides,
  };
}

function queryString(params = currentParams()) {
  return new URLSearchParams(params).toString();
}

async function refreshWindowedDashboard() {
  const limit = currentLimit();
  const requestId = ++state.windowRequestId;
  const params = currentParams({ window_size: limit });
  const qs = queryString(params);
  setStatus(`Recomputing charts for window ${limit}, seed ${params.seed}, ${params.fusion_mode} fusion...`);
  const [research, chartData, novelty, backend] = await Promise.all([
    getJson(`/api/research?${qs}`),
    getJson(`/api/charts?${qs}`),
    getJson(`/api/novelty?${qs}`),
    getJson("/api/backend/status"),
  ]);
  if (requestId !== state.windowRequestId) return;
  state.research = research;
  state.chartData = chartData;
  state.novelty = novelty;
  state.backend = backend;
  renderOverview();
  renderAnalysis();
  renderNovelty();
  renderBackendStatus();
  setStatus(`${research.limit.toLocaleString()} flows computed`);
}

async function refreshNoveltyForControls() {
  const limit = currentLimit();
  const alpha = currentAlpha();
  const qs = queryString(currentParams({ window_size: limit, alpha }));
  setStatus(`Refreshing reliability for alpha ${alpha.toFixed(2)}...`);
  state.novelty = await getJson(`/api/novelty?${qs}`);
  renderNovelty();
  setStatus(`Reliability refreshed for alpha ${state.novelty.alpha}`);
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

function metricScale(values, pad = 0.02) {
  const nums = values.map(Number).filter(Number.isFinite);
  if (!nums.length) return { min: 0, max: 1 };
  if (Math.min(...nums) === Math.max(...nums)) {
    return { min: Math.max(0, nums[0] - pad), max: Math.min(1, nums[0] + pad) };
  }
  return {
    min: Math.max(0, Math.min(...nums) - pad),
    max: Math.min(1, Math.max(...nums) + pad),
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
          label: "Baseline MLP",
          data: r.metrics.existing,
          backgroundColor: "rgba(244,183,64,.72)",
          borderColor: colors.amber,
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: "Neuro-symbolic",
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
      scales: { ...chartScales(), y: { ...metricScale([...r.metrics.existing, ...r.metrics.proposed]), grid: { color: "rgba(255,255,255,.05)" } } },
      plugins: { tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${pct(ctx.raw, 2)}` } } },
    },
  });

  makeChart("distributionChart", "chart-distribution", {
    type: "bar",
    data: {
      labels: r.class_distribution.labels,
      datasets: [
        {
          label: "Baseline labels",
          data: r.class_distribution.baseline_values || r.class_distribution.values,
          backgroundColor: "rgba(244,183,64,.55)",
          borderColor: colors.amber,
          borderWidth: 1,
          borderRadius: 5,
        },
        {
          label: "Proposed labels",
          data: r.class_distribution.proposed_values || r.class_distribution.values,
          backgroundColor: "rgba(25,211,197,.62)",
          borderColor: colors.cyan,
          borderWidth: 1,
          borderRadius: 5,
        },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: chartScales() },
  });
  renderImpactProof();
  renderArchitectureTelemetry();
  renderBackendStatus();
}

function renderImpactProof() {
  const r = state.research;
  if (!r || !r.rule_analytics) return;
  const analytics = r.rule_analytics;
  const proof = r.novelty_proof || {};
  $("#impact-verdict").textContent = proof.verdict === "proven" ? "proven" : "needs larger window";
  $("#impact-verdict").classList.toggle("good", proof.verdict === "proven");
  $("#impact-trigger-rate").textContent = pct(analytics.rule_trigger_rate || 0);
  $("#impact-trigger-count").textContent = `${analytics.rule_trigger_count || 0} rule firings`;
  $("#impact-change-rate").textContent = pct(analytics.prediction_change_rate || 0);
  $("#impact-change-count").textContent = `${analytics.prediction_change_count || 0} changed predictions`;
  $("#impact-delta-accuracy").textContent = signedPct(analytics.delta_accuracy || 0);
  $("#impact-delta-f1").textContent = signedPct(analytics.delta_f1 || 0);
  $("#impact-attack-recall").textContent = signedPct(analytics.binary_attack_recall_delta || 0);
  $("#impact-fn-rescues").textContent = analytics.false_negative_attack_rescues || 0;
  const examples = proof.examples || [];
  $("#impact-examples").innerHTML = examples.length ? examples.map(item => `
    <div class="example-item ${item.exact_correction ? "exact" : ""}">
      <strong>Sample ${item.sample}: ${item.mlp_label} -> ${item.neuro_symbolic_label}</strong>
      <span>true=${item.true_label} | ${item.rule_id} | strength=${Number(item.rule_strength || 0).toFixed(3)}</span>
      <p>${item.explanation || "Rule explanation unavailable."}</p>
    </div>
  `).join("") : `<div class="example-item"><strong>No correction examples in this window</strong><span>Increase the sample window to inspect more rare failure regions.</span></div>`;
}

function renderAnalysis() {
  const r = state.research;
  const c = state.chartData;
  $("#cache-status").textContent = `${r.limit.toLocaleString()} flows computed`;
  $("#sample-window-value").textContent = r.limit;
  $("#sample-window").value = r.limit;
  if (c.debug) console.info("Chart recompute debug", c.debug);

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
    type: "bar",
    data: {
      labels: c.improvement_curve.labels,
      datasets: [
        {
          type: "line",
          label: "Baseline macro F1",
          data: c.improvement_curve.existing_f1,
          borderColor: colors.amber,
          backgroundColor: "rgba(244,183,64,.1)",
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          tension: 0.25,
          yAxisID: "y",
        },
        {
          type: "line",
          label: "Neuro-symbolic macro F1",
          data: c.improvement_curve.proposed_f1,
          borderColor: colors.cyan,
          backgroundColor: "rgba(25,211,197,.12)",
          borderWidth: 3,
          pointRadius: 5,
          pointHoverRadius: 7,
          tension: 0.25,
          yAxisID: "y",
        },
        {
          type: "bar",
          label: "Attack recall lift (pts)",
          data: c.improvement_curve.attack_recall_delta_points,
          backgroundColor: "rgba(74,222,128,.35)",
          borderColor: colors.green,
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Evaluation window size" }, grid: { color: "rgba(255,255,255,.05)" } },
        y: { ...metricScale([...c.improvement_curve.existing_f1, ...c.improvement_curve.proposed_f1], 0.004), title: { display: true, text: "Macro F1" }, grid: { color: "rgba(255,255,255,.05)" } },
        y1: { position: "right", beginAtZero: true, title: { display: true, text: "Attack recall lift (pts)" }, grid: { drawOnChartArea: false }, ticks: { callback: value => `${value}` } },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => ctx.dataset.yAxisID === "y1"
              ? `${ctx.dataset.label}: ${Number(ctx.raw || 0).toFixed(3)}`
              : `${ctx.dataset.label}: ${pct(ctx.raw, 3)}`,
          },
        },
      },
    },
  });

  makeChart("perClassChart", "chart-per-class", {
    type: "radar",
    data: {
      labels: c.per_class.labels,
      datasets: [
        { label: "Baseline MLP F1", data: c.per_class.existing_f1, backgroundColor: "rgba(244,183,64,.14)", borderColor: colors.amber, borderWidth: 2 },
        { label: "Neuro-symbolic F1", data: c.per_class.proposed_f1, backgroundColor: "rgba(25,211,197,.18)", borderColor: colors.cyan, borderWidth: 2 },
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
    type: "bar",
    data: {
      labels: c.detection_counts.labels,
      datasets: [{ label: "Flows", data: c.detection_counts.values, backgroundColor: ["rgba(255,91,110,.72)", "rgba(244,183,64,.72)", "rgba(25,211,197,.72)", "rgba(155,135,255,.72)", "rgba(74,222,128,.72)"], borderColor: "#101720", borderWidth: 2, borderRadius: 5 }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: chartScales(), plugins: { legend: { display: false } } },
  });

  makeChart("errorRateChart", "chart-error-rate", {
    type: "bar",
    data: {
      labels: c.class_error_rate.labels,
      datasets: [
        { label: "Baseline error", data: c.class_error_rate.baseline_values || c.class_error_rate.values, backgroundColor: "rgba(244,183,64,.55)", borderColor: colors.amber, borderWidth: 1, borderRadius: 5 },
        { label: "Proposed error", data: c.class_error_rate.proposed_values || c.class_error_rate.values, backgroundColor: "rgba(255,91,110,.62)", borderColor: colors.red, borderWidth: 1, borderRadius: 5 },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { ...chartScales(), y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.05)" } } }, plugins: { tooltip: { callbacks: { label: ctx => pct(ctx.raw, 2) } } } },
  });

  makeChart("rocChart", "chart-roc", {
    type: "line",
    data: {
      datasets: [
        { label: `Baseline ROC AUC ${c.roc_curve.baseline?.auc ?? "n/a"}`, data: c.roc_curve.baseline?.points || c.roc_curve.points, borderColor: colors.amber, backgroundColor: "transparent", parsing: false, pointRadius: 0, borderDash: [7, 4], tension: 0.25 },
        { label: `Proposed ROC AUC ${c.roc_curve.proposed?.auc ?? c.roc_curve.auc ?? "n/a"}`, data: c.roc_curve.proposed?.points || c.roc_curve.points, borderColor: colors.green, backgroundColor: "rgba(74,222,128,.12)", parsing: false, pointRadius: 0, tension: 0.25, fill: true },
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

  makeChart("differenceChart", "chart-difference", {
    type: "bar",
    data: {
      labels: c.difference_chart.labels,
      datasets: [{
        label: "Proposed minus baseline",
        data: c.difference_chart.values,
        backgroundColor: c.difference_chart.values.map(value => Number(value) >= 0 ? "rgba(74,222,128,.65)" : "rgba(255,91,110,.65)"),
        borderColor: c.difference_chart.values.map(value => Number(value) >= 0 ? colors.green : colors.red),
        borderWidth: 1,
        borderRadius: 5,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: chartScales(), plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => signedPct(ctx.raw, 3) } } } },
  });

  makeChart("attackRecallGainChart", "chart-attack-recall-gain", {
    type: "bar",
    data: {
      labels: c.attack_recall_gain.labels,
      datasets: [
        { label: "Baseline recall", data: c.attack_recall_gain.baseline, backgroundColor: "rgba(244,183,64,.45)", borderColor: colors.amber, borderWidth: 1, borderRadius: 5 },
        { label: "Proposed recall", data: c.attack_recall_gain.proposed, backgroundColor: "rgba(25,211,197,.55)", borderColor: colors.cyan, borderWidth: 1, borderRadius: 5 },
        { type: "line", label: "Recall gain", data: c.attack_recall_gain.values, borderColor: colors.green, backgroundColor: "rgba(74,222,128,.12)", pointRadius: 4, yAxisID: "y1" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        ...chartScales(),
        y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.05)" } },
        y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { callback: value => `${Number(value).toFixed(2)}` } },
      },
    },
  });

  renderMatrix(r.classes, r.confusion_matrix);
  renderAuditTable(r.rows);
  renderImpactProof();
  renderArchitectureTelemetry();
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
    ["Learned rescue rules", b.symbolic_rule_summary?.count ?? "not loaded"],
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

function renderNovelty() {
  const n = state.novelty;
  if (!n) return;
  $("#novelty-ece").textContent = Number(n.calibration.ece || 0).toFixed(3);
  $("#novelty-coverage").textContent = pct(n.conformal.empirical_coverage || 0);
  $("#novelty-ood").textContent = pct(n.ood_drift.ood_rate || 0);
  $("#novelty-review").textContent = n.review_queue.length;

  makeChart("conformalChart", "chart-conformal", {
    type: "bar",
    data: {
      labels: ["Target coverage", "Empirical coverage", "Probability threshold"],
      datasets: [{
        label: `Alpha ${Number(n.alpha || 0).toFixed(2)}`,
        data: [
          n.conformal.target_coverage,
          n.conformal.empirical_coverage,
          n.conformal.probability_threshold,
        ],
        backgroundColor: ["rgba(75,163,255,.72)", "rgba(25,211,197,.72)", "rgba(244,183,64,.72)"],
        borderColor: [colors.blue, colors.cyan, colors.amber],
        borderWidth: 1,
        borderRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { ...chartScales(), y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.05)" } } },
      plugins: { legend: { display: true }, tooltip: { callbacks: { label: ctx => pct(ctx.raw, 2) } } },
    },
  });

  makeChart("calibrationChart", "chart-calibration", {
    type: "line",
    data: {
      labels: n.calibration.bins.map(row => row.bin),
      datasets: [
        {
          label: "Observed accuracy",
          data: n.calibration.bins.map(row => row.accuracy),
          borderColor: colors.cyan,
          backgroundColor: "rgba(25,211,197,.14)",
          tension: 0.35,
          fill: true,
        },
        {
          label: "Mean confidence",
          data: n.calibration.bins.map(row => row.confidence),
          borderColor: colors.amber,
          backgroundColor: "transparent",
          tension: 0.35,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { ...chartScales(), y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.05)" } } },
    },
  });

  makeChart("driftChart", "chart-drift", {
    type: "bar",
    data: {
      labels: n.ood_drift.top_drift_features.map(row => row.feature),
      datasets: [{
        label: "Mean |z|",
        data: n.ood_drift.top_drift_features.map(row => row.mean_abs_z),
        backgroundColor: "rgba(155,135,255,.68)",
        borderColor: colors.violet,
        borderWidth: 1,
        borderRadius: 5,
      }],
    },
    options: { responsive: true, maintainAspectRatio: false, scales: chartScales(), plugins: { legend: { display: false } } },
  });

  $("#novelty-review-body").innerHTML = n.review_queue.map(row => `
    <tr>
      <td>${row.idx}</td>
      <td>${row.true}</td>
      <td>${row.predicted}</td>
      <td>${pct(row.confidence, 1)}</td>
      <td>${Number(row.entropy || 0).toFixed(3)}</td>
      <td>${Number(row.ood_score || 0).toFixed(3)}</td>
      <td class="risk-${row.reason === "OOD" ? "attack" : "benign"}">${row.reason}</td>
    </tr>
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
      <td>${row.changed_prediction ? "yes" : "no"}</td>
      <td>${Number(row.rule_strength || 0).toFixed(3)}</td>
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

async function analyseFlow(index) {
  index = typeof index !== "undefined" ? index : Number($("#flow-index")?.value || 0);
  const result = await postJson("/api/defense/analyse", currentParams({ flow_index: index, idx: index }));
  const flow = result.flow;
  state.flow = flow;
  state.incident = result.incident;
  $("#flow-index").value = flow.index;
  renderDefense(flow, result.incident);
  setStatus(`Flow ${flow.index.toLocaleString()} analysed`);
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
    ["Changed prediction", flow.changed_prediction ? "yes" : "no"],
    ["Rule strength", Number(flow.rule_strength || 0).toFixed(3)],
    ["Explanation", flow.explanation || ""],
    ["Robust model", flow.robust_pred || "Unavailable"],
    ["Symbolic trace", rules],
    ...topFeatures,
  ].map(([key, value]) => `<div class="evidence-item"><span>${key}</span><strong>${value}</strong></div>`).join("");
  renderArchitectureTelemetry();
}

function renderArchitectureTelemetry() {
  if (!state.research) return;
  const windowEl = $("#arch-window");
  const rulesEl = $("#arch-rules");
  const changesEl = $("#arch-changes");
  if (!windowEl || !rulesEl || !changesEl) return;
  const analytics = state.research.rule_analytics || {};
  windowEl.textContent = Number(state.research.limit || 0).toLocaleString();
  rulesEl.textContent = Number(analytics.rule_trigger_count || 0).toLocaleString();
  changesEl.textContent = Number(analytics.prediction_change_count || 0).toLocaleString();
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
    code: "stream.normalize(window)",
  },
  {
    title: "Neural inference",
    copy: "The trained MLP estimates class probabilities across benign and attack families.",
    steps: ["Run baseline classifier", "Rank candidate attack classes", "Expose confidence distribution to the dashboard"],
    code: "mlp.predict_proba(flow)",
  },
  {
    title: "Symbolic reasoning",
    copy: "Domain rules correct or explain neural decisions using packet-rate, byte-rate, duration, and anomaly context.",
    steps: ["Check high-rate DDoS bursts", "Check slow sustained attacks", "Attach fired-rule trace to the prediction"],
    code: "rules.fuse(probabilities)",
  },
  {
    title: "Defence response",
    copy: "The final label is converted into practical containment guidance for analyst review.",
    steps: ["Warn user when attack is detected", "Recommend isolation, rate limiting, or blocking", "Export evidence for incident reporting"],
    code: "response.apply(playbook)",
  },
];

function initArchitectureScene() {
  const canvas = $("#architecture-canvas");
  if (!canvas || !window.THREE || state.architectureScene) return;

  const THREE = window.THREE;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  if ("outputColorSpace" in renderer && THREE.SRGBColorSpace) renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 5.8, 10.5);
  camera.lookAt(0, 0, 0);

  const root = new THREE.Group();
  root.rotation.y = -0.18;
  scene.add(root);

  const ambient = new THREE.AmbientLight(0x8fb8ff, 0.7);
  scene.add(ambient);
  const key = new THREE.DirectionalLight(0x9be8ff, 1.2);
  key.position.set(-3, 6, 8);
  scene.add(key);
  const rim = new THREE.PointLight(0x19d3c5, 2.2, 12);
  rim.position.set(3, 2, 4);
  scene.add(rim);

  const grid = new THREE.GridHelper(11, 22, 0x24526c, 0x123044);
  grid.position.y = -1.65;
  grid.material.transparent = true;
  grid.material.opacity = 0.38;
  root.add(grid);

  const positions = [
    new THREE.Vector3(-4.0, -0.9, 0.4),
    new THREE.Vector3(-1.35, 1.2, -0.65),
    new THREE.Vector3(1.7, -0.35, 0.35),
    new THREE.Vector3(4.1, 1.05, -0.55),
  ];
  const nodeColors = [0x4ba3ff, 0xf4b740, 0x19d3c5, 0x4ade80];
  const nodes = positions.map((position, index) => {
    const group = new THREE.Group();
    group.position.copy(position);
    const shell = new THREE.Mesh(
      new THREE.BoxGeometry(1.35, 0.68, 1.0),
      new THREE.MeshPhysicalMaterial({
        color: 0x0c1722,
        emissive: nodeColors[index],
        emissiveIntensity: 0.08,
        transparent: true,
        opacity: 0.68,
        roughness: 0.32,
        metalness: 0.38,
        transmission: 0.12,
      }),
    );
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(shell.geometry),
      new THREE.LineBasicMaterial({ color: nodeColors[index], transparent: true, opacity: 0.72 }),
    );
    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.28, 1),
      new THREE.MeshStandardMaterial({ color: nodeColors[index], emissive: nodeColors[index], emissiveIntensity: 0.45, roughness: 0.25 }),
    );
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.58, 0.012, 8, 72),
      new THREE.MeshBasicMaterial({ color: nodeColors[index], transparent: true, opacity: 0.48 }),
    );
    ring.rotation.x = Math.PI / 2.5;
    group.add(shell, edges, core, ring);
    root.add(group);
    return { group, shell, edges, core, ring, base: position.clone(), color: nodeColors[index] };
  });

  const curves = [];
  for (let i = 0; i < positions.length - 1; i += 1) {
    const start = positions[i];
    const end = positions[i + 1];
    const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, i % 2 === 0 ? 0.85 : -0.65, 0.35));
    const curve = new THREE.CatmullRomCurve3([start, mid, end]);
    curves.push(curve);
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 44, 0.018, 8, false),
      new THREE.MeshBasicMaterial({ color: i === 1 ? 0xf4b740 : 0x19d3c5, transparent: true, opacity: 0.42 }),
    );
    root.add(tube);
  }
  const fullPath = new THREE.CatmullRomCurve3([
    positions[0],
    positions[0].clone().lerp(positions[1], 0.5).add(new THREE.Vector3(0, 0.85, 0.35)),
    positions[1],
    positions[1].clone().lerp(positions[2], 0.5).add(new THREE.Vector3(0, -0.65, 0.35)),
    positions[2],
    positions[2].clone().lerp(positions[3], 0.5).add(new THREE.Vector3(0, 0.85, 0.35)),
    positions[3],
  ]);

  const particleGeometry = new THREE.SphereGeometry(0.055, 12, 12);
  const particles = Array.from({ length: 32 }, (_, index) => {
    const material = new THREE.MeshBasicMaterial({
      color: index % 5 === 0 ? 0xf4b740 : index % 3 === 0 ? 0x4ba3ff : 0x19d3c5,
      transparent: true,
      opacity: 0.92,
    });
    const particle = new THREE.Mesh(particleGeometry, material);
    root.add(particle);
    return { mesh: particle, offset: index / 32, speed: 0.035 + (index % 7) * 0.004 };
  });

  const resize = () => {
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvas);
  resize();

  let frame = 0;
  const clock = new THREE.Clock();
  const animate = () => {
    const elapsed = clock.getElapsedTime();
    root.rotation.y = -0.18 + Math.sin(elapsed * 0.18) * 0.08;
    nodes.forEach((node, index) => {
      const active = state.architectureScene?.activeStage === index;
      const pulse = active ? 1.0 + Math.sin(elapsed * 3.2) * 0.045 : 1.0;
      node.group.scale.setScalar(pulse);
      node.group.position.y = node.base.y + Math.sin(elapsed * 1.4 + index) * 0.055;
      node.ring.rotation.z += active ? 0.028 : 0.01;
      node.core.rotation.x += 0.012 + index * 0.001;
      node.core.rotation.y += 0.018;
    });
    particles.forEach(item => {
      const progress = (item.offset + elapsed * item.speed) % 1;
      item.mesh.position.copy(fullPath.getPointAt(progress));
      item.mesh.material.opacity = 0.35 + Math.sin(progress * Math.PI) * 0.58;
    });
    renderer.render(scene, camera);
    frame = requestAnimationFrame(animate);
  };

  state.architectureScene = { renderer, scene, camera, root, nodes, particles, resizeObserver, activeStage: 0, animationFrame: frame };
  updateArchitectureSceneStage(0);
  animate();
}

function updateArchitectureSceneStage(stage) {
  const arch = state.architectureScene;
  if (!arch) return;
  arch.activeStage = stage;
  arch.nodes.forEach((node, index) => {
    const active = index === stage;
    node.shell.material.emissiveIntensity = active ? 0.48 : 0.08;
    node.shell.material.opacity = active ? 0.86 : 0.58;
    node.edges.material.opacity = active ? 1 : 0.52;
    node.core.material.emissiveIntensity = active ? 1.2 : 0.42;
    node.ring.material.opacity = active ? 0.95 : 0.42;
  });
}

function setArchitectureStage(stage) {
  $$(".arch-step").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.stage) === stage));
  $$(".node").forEach((node, index) => node.classList.toggle("active", index === stage));
  $("#stage-title").textContent = stageCopy[stage].title;
  $("#stage-copy").textContent = stageCopy[stage].copy;
  $("#stage-steps").innerHTML = stageCopy[stage].steps.map(step => `<li>${step}</li>`).join("");
  $("#arch-terminal-stage").textContent = stageCopy[stage].title;
  $("#arch-terminal-code").textContent = stageCopy[stage].code;
  const scene = $("#scene");
  if (scene) scene.dataset.stage = stage;
  updateArchitectureSceneStage(stage);
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

function renderedChartPayload() {
  return Object.entries(state.charts)
    .map(([name, chart]) => {
      if (!chart || !chart.canvas) return null;
      chart.resize();
      chart.update("none");
      const image = chart.toBase64Image("image/png", 1);
      if (!image || image === "data:,") return null;
      return { name, image };
    })
    .filter(Boolean);
}

async function exportAllChartsToProject() {
  const charts = renderedChartPayload();
  if (!charts.length) throw new Error("No rendered charts are available to export yet.");

  const result = await postJson("/api/export-charts", {
    charts,
    metadata: {
      window: currentLimit(),
      alpha: currentAlpha(),
      flow_index: state.flow?.index ?? null,
    },
  });
  $("#run-all-feedback").textContent = `Saved ${result.saved.length} charts`;
  setStatus(`Saved ${result.saved.length} charts to ${result.export_dir}`);
}

function setupExports() {
  $("#btn-export-json").addEventListener("click", () => {
    download("neuro_symbolic_dashboard_export.json", JSON.stringify({ overview: state.overview, research: state.research, flow: state.flow }, null, 2), "application/json");
  });
  bindAsyncClick("#btn-export-all-charts", "Exporting charts", exportAllChartsToProject);
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
  bindAsyncClick("#btn-refresh", "Refreshing dashboard", loadDashboard);
  bindAsyncClick("#btn-run-all", "Running full pipeline", runAll);
  bindAsyncClick("#btn-backend-refresh", "Refreshing backend status", async () => {
    state.backend = await getJson("/api/backend/status");
    renderBackendStatus();
  });
  // Interactive slider: update display immediately and debounce server recompute
  const sampleEl = $("#sample-window");
  if (sampleEl) {
    let debounceTimer = null;
    sampleEl.addEventListener("input", event => {
      const val = event.target.value;
      $("#sample-window-value").textContent = val;
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(async () => {
        try {
          await refreshWindowedDashboard();
        } catch (err) {
          showActionError(err);
        }
      }, DEBOUNCE_MS);
    });
    sampleEl.addEventListener("change", async event => {
      if (debounceTimer) clearTimeout(debounceTimer);
      $("#sample-window-value").textContent = event.target.value;
      try {
        await refreshWindowedDashboard();
      } catch (err) {
        showActionError(err);
      }
    });
  }

  bindAsyncClick("#btn-refresh-novelty", "Refreshing reliability evidence", async () => {
    await refreshNoveltyForControls();
  });

  const alphaEl = $("#novelty-alpha");
  if (alphaEl) {
    let alphaTimer = null;
    const syncBeta = () => {
      const betaEl = $("#fusion-beta");
      if (betaEl) betaEl.value = Number((1 - currentAlpha()).toFixed(2));
    };
    alphaEl.addEventListener("input", () => {
      syncBeta();
      if (alphaTimer) clearTimeout(alphaTimer);
      alphaTimer = setTimeout(() => refreshWindowedDashboard().catch(showActionError), DEBOUNCE_MS);
    });
    alphaEl.addEventListener("change", () => {
      syncBeta();
      if (alphaTimer) clearTimeout(alphaTimer);
      refreshWindowedDashboard().catch(showActionError);
    });
  }

  ["#fusion-beta", "#seed-selector", "#fusion-mode"].forEach(selector => {
    const el = $(selector);
    if (!el) return;
    let timer = null;
    const eventName = el.type === "checkbox" ? "change" : "input";
    el.addEventListener(eventName, () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => refreshWindowedDashboard().catch(showActionError), DEBOUNCE_MS);
    });
    if (eventName !== "change") {
      el.addEventListener("change", () => refreshWindowedDashboard().catch(showActionError));
    }
  });

  const flowIndexEl = $("#flow-index");
  if (flowIndexEl) {
    let flowTimer = null;
    flowIndexEl.addEventListener("change", async () => {
      if (flowTimer) clearTimeout(flowTimer);
      try {
        await analyseFlow();
        await refreshWindowedDashboard();
      } catch (err) {
        showActionError(err);
      }
    });
  }

  bindAsyncClick("#btn-analyse-flow", "Analysing flow", async () => {
    await analyseFlow();
    await refreshWindowedDashboard();
  });
  bindAsyncClick("#btn-random-flow", "Selecting random flow", () => {
    const current = Number($("#flow-index")?.value || 0);
    if (state.maxIndex <= 0) return analyseFlow(0).then(refreshWindowedDashboard);
    let next = Math.floor(Math.random() * state.maxIndex);
    if (next >= current) next += 1;
    return analyseFlow(Math.min(next, state.maxIndex)).then(refreshWindowedDashboard);
  });
  bindAsyncClick("#btn-contain-flow", "Applying simulated containment", async () => {
    if (!state.incident) return;
    const result = await postJson("/api/defense/contain", { incident_id: state.incident.incident_id });
    state.incident = result.incident;
    $("#decision-action").textContent = result.message;
    renderIncident(result.incident);
  });
  bindAsyncClick("#btn-defense-status", "Loading defense status", async () => {
    const status = await getJson("/api/defense/status");
    $("#cache-status").textContent = `${status.total_incidents} backend incidents tracked`;
  });
  $$(".arch-step").forEach(btn => btn.addEventListener("click", () => setArchitectureStage(Number(btn.dataset.stage))));
}

async function loadDashboard() {
  $("#cache-status").textContent = "Loading research cache";
  const qs = queryString();
  const [overview, research, chartData, novelty, backend] = await Promise.all([
    getJson("/api/overview"),
    getJson(`/api/research?${qs}`),
    getJson(`/api/charts?${qs}`),
    getJson(`/api/novelty?${qs}`),
    getJson("/api/backend/status"),
  ]);
  state.overview = overview;
  state.research = research;
  state.chartData = chartData;
  state.novelty = novelty;
  state.backend = backend;
  renderOverview();
  renderAnalysis();
  renderNovelty();
  await analyseFlow(0);
  setArchitectureStage(0);
  startArchitectureLoop();
  $("#cache-status").textContent = `${research.limit.toLocaleString()} flows computed`;
}

async function runAll() {
  const button = $("#btn-run-all");
  const feedback = $("#run-all-feedback");
  button.classList.add("loading");
  feedback.textContent = "Recomputing...";
  try {
    const result = await postJson("/api/run-all", currentParams({ limit: currentLimit(), flow_idx: currentFlowIndex() }));
    state.overview = result.overview;
    state.research = result.research;
    state.chartData = result.charts;
    state.novelty = result.novelty;
    state.backend = result.backend;
    state.flow = result.defense.flow;
    state.incident = result.defense.incident;
    state.lastRunDebug = result.debug;
    renderOverview();
    renderAnalysis();
    renderNovelty();
    renderDefense(state.flow, state.incident);
    renderBackendStatus();
    feedback.textContent = `Run All complete in ${Number(result.debug?.api_output_summary?.elapsed_ms || 0).toFixed(0)} ms`;
    setStatus(`Run All recomputed ${state.research.limit.toLocaleString()} flows`);
    console.info("Run All debug", result.debug);
  } finally {
    button.classList.remove("loading");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupControls();
  setupExports();
  initArchitectureScene();
  loadDashboard().catch(error => {
    showActionError(error);
    setStatus("Dashboard data failed to load");
  });
});
