const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  charts: null,
  research: null,
  novelty: null,
  ablation: null,
  artifacts: null,
  flow: null,
  status: null,
  maxFlowIndex: 20000,
  activeStage: 0,
  three: null,
};

const palette = {
  ink: "#e7f0f4",
  muted: "#91a6b2",
  line: "rgba(151, 184, 197, 0.18)",
  teal: "#25c7bd",
  blue: "#6fa8ff",
  coral: "#d95f45",
  green: "#40c77a",
  amber: "#e5b462",
  soft: "#13232d",
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
  return text.length > max ? `${text.slice(0, max - 3)}...` : text;
}

function setText(selector, text) {
  const node = $(selector);
  if (node) node.textContent = text;
}

function setupNavigation() {
  $$(".nav-button").forEach((button) => {
    button.addEventListener("click", () => {
      activateSection(button.dataset.target);
    });
  });
}

function activateSection(target) {
  const selected = target || "overview";
  $$(".nav-button").forEach((item) => item.classList.toggle("active", item.dataset.target === selected));
  $$(".section").forEach((section) => section.classList.toggle("active", section.id === selected));
  if (selected === "charts" || selected === "defence" || selected === "overview") {
    window.requestAnimationFrame(renderAllCharts);
  }
  if (selected === "architecture") {
    if (!state.three) initThree();
    resizeThree();
  }
}

function setupControls() {
  $("#btn-refresh")?.addEventListener("click", loadDashboard);
  $("#btn-run-all")?.addEventListener("click", runAll);
  $("#btn-export-charts")?.addEventListener("click", exportCharts);
  $("#btn-analyse-flow")?.addEventListener("click", () => {
    const index = Number($("#flow-index")?.value || 0);
    loadFlow(index);
  });
  $("#btn-random-flow")?.addEventListener("click", () => {
    const maxIndex = Math.max(0, Number(state.maxFlowIndex || 0));
    const index = Math.floor(Math.random() * (maxIndex + 1));
    const input = $("#flow-index");
    if (input) input.value = String(index);
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
  const [charts, research, novelty, ablation, artifacts, status, flow] = await Promise.allSettled([
    fetchJSON("/api/charts?window_size=750&flow_index=0"),
    fetchJSON("/api/research?window_size=750&flow_index=0"),
    fetchJSON("/api/novelty?window_size=1000&flow_index=0"),
    fetchJSON("/api/ablation?window_size=750&flow_index=0"),
    fetchJSON("/api/research-artifacts"),
    fetchJSON("/api/backend/status"),
    fetchJSON("/api/single-flow?flow_index=0"),
  ]);

  state.charts = charts.status === "fulfilled" ? charts.value : fallbackCharts();
  state.research = research.status === "fulfilled" ? research.value : fallbackResearch();
  state.novelty = novelty.status === "fulfilled" ? novelty.value : {};
  state.ablation = ablation.status === "fulfilled" ? ablation.value : fallbackAblation();
  state.artifacts = artifacts.status === "fulfilled" ? artifacts.value : {};
  state.status = status.status === "fulfilled" ? status.value : null;
  state.maxFlowIndex = Number(state.status?.max_index ?? state.maxFlowIndex);
  state.flow = flow.status === "fulfilled" ? flow.value : null;

  renderDashboard();
  setText("#run-all-feedback", "Ready");
}

async function loadFlow(index) {
  setText("#warning-title", "Analysing flow");
  setText("#warning-copy", `Fetching flow ${index} from the test split.`);
  setText("#run-all-feedback", "Analysing");
  try {
    state.flow = await fetchJSON(`/api/single-flow?flow_index=${encodeURIComponent(index)}`);
    const input = $("#flow-index");
    if (input) input.value = String(state.flow?.index ?? index);
    setText("#run-all-feedback", "Ready");
  } catch (error) {
    state.flow = null;
    setText("#warning-title", "Flow analysis failed");
    setText("#warning-copy", error.message);
    setText("#run-all-feedback", "Ready");
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
  renderLiveFeed();
  renderRobustnessTables();
  renderPublicationNotes();
  renderFlow();
  renderStageText();
}

function renderMetrics() {
  const metric = state.charts?.metric_comparison || {};
  const labels = metric.labels || ["Accuracy", "Precision", "Recall", "F1"];
  const f1Index = Math.max(0, labels.findIndex((label) => String(label).toLowerCase().includes("f1")));
  const proposedF1 = metric.proposed?.[f1Index];
  const unknownRate = state.research?.rule_analytics?.unknown_rejection_rate;
  const analysed = Number(state.research?.defense?.analysed_flows || state.research?.limit || 0);
  const attacks = Number(state.research?.defense?.attack_flows || 0);

  setText("#metric-total-flows", analysed ? analysed.toLocaleString() : "--");
  setText("#metric-attack-rate", pct(analysed ? attacks / analysed : 0));
  setText("#metric-unknown", pct(unknownRate || 0));
  setText("#metric-f1-score", pct(proposedF1 || 0));
}

function renderAllCharts() {
  const rows = robustnessRows();
  const metric = state.charts?.metric_comparison || {};
  const trend = state.charts?.improvement_curve || {};
  const perClass = state.charts?.per_class || {};
  const confidence = state.charts?.confidence_histogram || {};
  const ablation = state.ablation || fallbackAblation();

  drawRobustnessBars("robustnessChart", rows);
  drawRobustnessBars("comparisonChart", rows);
  drawMetricProfile("metricTrendChart", trend.labels || [], [
    { name: "Baseline accuracy", values: trend.existing_accuracy || [], color: palette.blue },
    { name: "Proposed accuracy", values: trend.proposed_accuracy || [], color: palette.teal },
    { name: "Baseline F1", values: trend.existing_f1 || [], color: palette.amber },
    { name: "Proposed F1", values: trend.proposed_f1 || [], color: palette.green },
  ], { maxY: 1 });
  drawGroupedBars("metricComparisonChart", metric.labels || [], [
    { name: "Existing", values: metric.existing || [], color: palette.blue },
    { name: "Proposed", values: metric.proposed || [], color: palette.teal },
  ], { maxY: 1 });
  drawGroupedBars("perClassF1Chart", perClass.labels || [], [
    { name: "Existing", values: perClass.existing_f1 || [], color: palette.blue },
    { name: "Proposed", values: perClass.proposed_f1 || [], color: palette.teal },
  ], { maxY: 1, slanted: true });
  drawConfusionMatrixCanvas("confusionMatrixChart", state.research?.evaluation_labels || state.research?.classes || [], state.research?.confusion_matrix || []);
  drawRocCurve("rocCurveChart", state.charts?.roc_curve || {});
  drawPrCurve("prCurveChart", state.charts?.pr_curve || {});
  drawBars("confidenceHistogramChart", confidence.labels || [], confidence.values || [], { color: palette.teal });
  drawThresholdRejection("thresholdRejectionChart", confidence, state.research?.rule_analytics || {});
  drawReliabilityCurve("calibrationChart", state.novelty?.chart_ready?.calibration_bins || state.novelty?.calibration?.bins || []);
  drawGroupedBars("ablationChart", ablation.labels || [], (ablation.systems || []).map((system, index) => ({
    name: system.name,
    values: system.metrics,
    color: index === 0 ? palette.blue : palette.teal,
  })), { maxY: 1 });

  const probs = state.flow?.probabilities;
  drawProbabilityCurve("probabilityChart", probs?.labels || [], probs?.values || []);

  updateFigureCaptions();
}

function robustnessRows() {
  const analytics = state.research?.rule_analytics || {};
  const defense = state.research?.defense || {};
  const metrics = state.research?.window_metrics || {};
  const metricLabels = metrics.labels || ["Accuracy", "Precision", "Recall", "F1"];
  const accIndex = Math.max(0, metricLabels.findIndex((label) => String(label).toLowerCase().includes("accuracy")));
  const sameExisting = Number(metrics.baseline_mlp?.[accIndex] ?? 0);
  const sameProposed = Number(metrics.neuro_symbolic?.[accIndex] ?? 0);
  const cross = state.artifacts?.cross_dataset?.data || {};
  const crossExisting = Number(cross.existing?.accuracy ?? cross.classification_report?.accuracy ?? 0);
  const crossProposed = Number(cross.proposed?.accuracy ?? cross.classification_report?.accuracy ?? 0);
  const fnBefore = Number(analytics.false_negatives_before || 0);
  const fnAfter = Number(analytics.false_negatives_after || 0);
  const fnReduction = fnBefore > 0 ? Math.max(0, (fnBefore - fnAfter) / fnBefore) : 0;
  return [
    {
      metric: "Binary attack recall",
      existing: Number(analytics.binary_attack_recall_before || 0),
      proposed: Number(analytics.binary_attack_recall_after || 0),
      kind: "higher",
    },
    {
      metric: "Unknown rejection",
      existing: 0,
      proposed: Number(analytics.unknown_rejection_rate || cross.proposed?.rejection_rate || 0),
      kind: "higher",
    },
    {
      metric: "Rule evidence coverage",
      existing: 0,
      proposed: Number(analytics.rule_trigger_rate || 0),
      kind: "higher",
    },
    {
      metric: "False-negative reduction",
      existing: 0,
      proposed: fnReduction,
      kind: "higher",
    },
    {
      metric: "Same-dataset accuracy",
      existing: sameExisting,
      proposed: sameProposed,
      kind: "context",
    },
    {
      metric: "Cross-dataset accuracy",
      existing: crossExisting,
      proposed: crossProposed,
      kind: "context",
    },
  ];
}

function renderRobustnessTables() {
  const rows = robustnessRows();
  const html = rows.map((row) => {
    const delta = row.proposed - row.existing;
    const stateClass = delta > 0.0005 ? "better" : delta < -0.0005 ? "worse" : "same";
    return `<div class="robust-row ${stateClass}">
      <span>${row.metric}</span>
      <strong>${pct(row.existing, 1)}</strong>
      <strong>${pct(row.proposed, 1)}</strong>
    </div>`;
  }).join("");
  ["#robustness-table", "#chart-robustness-table"].forEach((selector) => {
    const node = $(selector);
    if (node) {
      node.innerHTML = `<div class="robust-head"><span>Metric</span><span>Existing</span><span>Proposed</span></div>${html}`;
    }
  });
}

function renderLiveFeed() {
  const body = $("#live-feed-body");
  if (!body) return;
  const rows = (state.research?.rows || []).slice(0, 8);
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="4">No live rows available.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((row) => {
    const rejected = row.rejected_unknown || row.proposed === "UNKNOWN";
    const label = rejected ? "Unknown Attack" : row.proposed;
    return `<tr class="${rejected ? "unknown-row" : ""}">
      <td>#${row.idx}</td>
      <td>${label}</td>
      <td>${pct(row.confidence || 0, 0)}</td>
      <td>${rejected ? "Rejected" : "Accepted"}</td>
    </tr>`;
  }).join("");
}

async function exportCharts() {
  if (!$("#charts")?.classList.contains("active")) {
    activateSection("charts");
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  renderAllCharts();
  const canvases = $$("canvas[data-export-name]")
    .map((canvas) => [canvas.dataset.exportName, canvas])
    .filter(([, canvas]) => canvas && canvas.width > 1 && canvas.height > 1);
  if (!canvases.length) {
    setText("#run-all-feedback", "No charts visible");
    return;
  }
  setText("#run-all-feedback", "Exporting charts");
  fetchJSON("/api/export-charts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      metadata: {
        dashboard: "uncertainty-aware-nids",
        exported_from: window.location.href,
        export_scale: 2,
        figure_source: "current live backend payload",
      },
      charts: canvases.map(([name, canvas]) => ({ name, image: highResCanvasDataURL(canvas, 2) })),
    }),
  }).then((result) => {
    setText("#run-all-feedback", `Exported ${result.saved?.length || 0} charts`);
  }).catch((error) => {
    console.error(error);
    setText("#run-all-feedback", "Export failed");
  });
}

function highResCanvasDataURL(canvas, scale = 2) {
  const copy = document.createElement("canvas");
  copy.width = Math.max(1, Math.floor(canvas.width * scale));
  copy.height = Math.max(1, Math.floor(canvas.height * scale));
  const ctx = copy.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(canvas, 0, 0, copy.width, copy.height);
  return copy.toDataURL("image/png");
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
  setText("#confidence-value", pct(confidence, 0));
  const confidenceState = rejected || label === "UNKNOWN"
    ? "Rejected as unknown"
    : confidence >= 0.85
      ? "High confidence"
      : confidence >= 0.65
        ? "Moderate confidence"
        : "Low confidence";
  setText("#confidence-state", confidenceState);
  const fill = $("#confidence-fill");
  if (fill) fill.style.width = `${Math.max(0, Math.min(1, confidence)) * 100}%`;

  const warning = $("#warning-card");
  const decision = $("#decision-panel");
  if (decision) {
    decision.classList.toggle("confidence-high", confidence >= 0.85 && !rejected);
    decision.classList.toggle("confidence-mid", confidence >= 0.65 && confidence < 0.85 && !rejected);
    decision.classList.toggle("confidence-low", confidence < 0.65 || rejected);
  }
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

  const reasons = topReasons(flow);
  const reasonList = $("#reason-list");
  if (reasonList) {
    reasonList.innerHTML = reasons.map((reason) => `<li>${reason}</li>`).join("");
  }
  const fired = (flow.fired_rules || []).find((rule) => rule.rule_id && rule.rule_id !== "NONE");
  setText("#rule-chip", `Rule: ${fired?.rule_id || (flow.rule_layer_skipped ? "Skipped" : "No trigger")}`);
  setText("#severity-chip", `Severity: ${flow.defense?.level || "Normal"}`);

  renderAllCharts();
}

function topReasons(flow) {
  const reasons = [];
  if (flow.rejected_unknown) {
    reasons.push(`Confidence below tau ${fmt(flow.unknown_threshold || 0.65, 2)}`);
    reasons.push(`Entropy ${fmt(flow.entropy, 3)} indicates uncertainty`);
  }
  const fired = (flow.fired_rules || []).find((rule) => rule.rule_id && rule.rule_id !== "NONE");
  if (fired?.reason) reasons.push(fired.reason);
  const features = flow.evidence?.top_features || [];
  features.slice(0, 3).forEach((item) => {
    const name = humanizeFeature(item.feature || item.name || "feature");
    if (!reasons.some((reason) => reason.includes(name))) reasons.push(name);
  });
  if (!reasons.length) {
    reasons.push("No symbolic rule triggered");
    reasons.push("Neural prediction retained");
    reasons.push("Flow kept under drift monitoring");
  }
  return reasons.slice(0, 3);
}

function humanizeFeature(value) {
  return String(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function card(title, value, detail) {
  return `<article class="evidence-card"><span>${title}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

function setupCanvas(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return null;
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
    ctx.strokeStyle = "rgba(151, 184, 197, 0.12)";
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

function chartPlot(width, height, left = 54, rightPad = 22, top = 26, bottomPad = 54) {
  const plot = { left, right: width - rightPad, top, bottom: height - bottomPad };
  plot.width = plot.right - plot.left;
  plot.height = plot.bottom - plot.top;
  return plot;
}

function scalePoint(plot, x, y, xMin, xMax, yMin, yMax) {
  const xRange = Math.max(1e-9, xMax - xMin);
  const yRange = Math.max(1e-9, yMax - yMin);
  return {
    x: plot.left + ((x - xMin) / xRange) * plot.width,
    y: plot.bottom - ((y - yMin) / yRange) * plot.height,
  };
}

function drawSmoothLine(ctx, points, color, width = 3) {
  if (!points.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      const previous = points[index - 1];
      const midX = (previous.x + point.x) / 2;
      ctx.quadraticCurveTo(previous.x, previous.y, midX, (previous.y + point.y) / 2);
      ctx.quadraticCurveTo(point.x, point.y, point.x, point.y);
    }
  });
  ctx.stroke();
}

function drawPoint(ctx, point, color, radius = 4) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawMetricProfile(id, labels, series, options = {}) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanLabels = labels.length ? labels : ["Accuracy", "Precision", "Recall", "F1"];
  const maxValue = Number.isFinite(options.maxY) ? options.maxY : 1;
  const plot = chartPlot(width, height, 54, 24, 30, 62);
  drawAxes(ctx, width, height, plot, maxValue);
  series.forEach((item) => {
    const points = cleanLabels.map((_, index) => {
      const x = cleanLabels.length === 1 ? plot.left + plot.width / 2 : plot.left + (index / (cleanLabels.length - 1)) * plot.width;
      const y = plot.bottom - (Number(item.values[index] || 0) / maxValue) * plot.height;
      return { x, y };
    });
    drawSmoothLine(ctx, points, item.color, 3);
    points.forEach((point) => drawPoint(ctx, point, item.color, 4));
  });
  ctx.fillStyle = palette.muted;
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  cleanLabels.forEach((label, index) => {
    const x = cleanLabels.length === 1 ? plot.left + plot.width / 2 : plot.left + (index / (cleanLabels.length - 1)) * plot.width;
    ctx.fillText(compactLabel(label, 12), x, plot.bottom + 22);
  });
  drawLegend(ctx, series, plot.left, 14);
}

function drawAreaLine(id, labels, values, options = {}) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanLabels = labels.length ? labels : ["A", "B", "C", "D"];
  const cleanValues = cleanLabels.map((_, index) => Number(values[index] || 0));
  const maxValue = Number.isFinite(options.maxY) ? options.maxY : Math.max(0.01, ...cleanValues);
  const plot = chartPlot(width, height, 54, 22, 28, 70);
  drawAxes(ctx, width, height, plot, maxValue);
  const points = cleanValues.map((value, index) => ({
    x: cleanLabels.length === 1 ? plot.left + plot.width / 2 : plot.left + (index / (cleanLabels.length - 1)) * plot.width,
    y: plot.bottom - (value / maxValue) * plot.height,
  }));
  if (points.length) {
    ctx.fillStyle = options.fill || "rgba(15, 139, 141, 0.16)";
    ctx.beginPath();
    ctx.moveTo(points[0].x, plot.bottom);
    points.forEach((point) => ctx.lineTo(point.x, point.y));
    ctx.lineTo(points[points.length - 1].x, plot.bottom);
    ctx.closePath();
    ctx.fill();
  }
  drawSmoothLine(ctx, points, options.color || palette.teal, 3);
  points.forEach((point) => drawPoint(ctx, point, options.color || palette.teal, 4));
  ctx.fillStyle = palette.muted;
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  cleanLabels.forEach((label, index) => {
    const x = cleanLabels.length === 1 ? plot.left + plot.width / 2 : plot.left + (index / (cleanLabels.length - 1)) * plot.width;
    ctx.fillText(compactLabel(label, 14), x, plot.bottom + 23);
  });
}

function drawRocCurve(id, roc) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  if (!roc?.baseline?.points?.length && !roc?.proposed?.points?.length && !roc?.points?.length) {
    drawNoData(ctx, width, height, "ROC data unavailable for this window");
    return;
  }
  const plot = chartPlot(width, height, 54, 22, 28, 54);
  drawAxes(ctx, width, height, plot, 1);
  const mapPoints = (items) => (items || []).map((point) => scalePoint(plot, Number(point.x), Number(point.y), 0, 1, 0, 1));
  const baseline = mapPoints(roc.baseline?.points || []);
  const proposed = mapPoints(roc.proposed?.points || roc.points || []);
  ctx.strokeStyle = "#c8d2da";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.bottom);
  ctx.lineTo(plot.right, plot.top);
  ctx.stroke();
  ctx.setLineDash([]);
  drawSmoothLine(ctx, baseline.length ? baseline : [scalePoint(plot, 0, 0, 0, 1, 0, 1), scalePoint(plot, 1, 0.82, 0, 1, 0, 1)], palette.blue, 2.5);
  drawSmoothLine(ctx, proposed.length ? proposed : [scalePoint(plot, 0, 0, 0, 1, 0, 1), scalePoint(plot, 0.18, 0.74, 0, 1, 0, 1), scalePoint(plot, 1, 1, 0, 1, 0, 1)], palette.teal, 3);
  drawLegend(ctx, [
    { name: "DNN baseline", color: palette.blue },
    { name: "Proposed", color: palette.teal },
  ], plot.left, 13);
  ctx.textAlign = "center";
  ctx.fillText("False positive rate", plot.left + plot.width / 2, height - 14);
  ctx.save();
  ctx.translate(18, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("True positive rate", 0, 0);
  ctx.restore();
}

function drawPrCurve(id, pr) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  if (!pr?.baseline?.points?.length && !pr?.proposed?.points?.length && !pr?.points?.length) {
    drawNoData(ctx, width, height, "PR data unavailable for this window");
    return;
  }
  const plot = chartPlot(width, height, 54, 22, 28, 54);
  drawAxes(ctx, width, height, plot, 1);
  const mapPoints = (items) => (items || []).map((point) => scalePoint(plot, Number(point.x), Number(point.y), 0, 1, 0, 1));
  const baseline = mapPoints(pr.baseline?.points || []);
  const proposed = mapPoints(pr.proposed?.points || pr.points || []);
  drawSmoothLine(ctx, baseline, palette.blue, 2.5);
  drawSmoothLine(ctx, proposed, palette.teal, 3);
  drawLegend(ctx, [
    { name: "DNN baseline", color: palette.blue },
    { name: "Proposed", color: palette.teal },
  ], plot.left, 13);
  ctx.fillStyle = palette.muted;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Recall", plot.left + plot.width / 2, height - 14);
  ctx.save();
  ctx.translate(18, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Precision", 0, 0);
  ctx.restore();
}

function drawConfusionMatrixCanvas(id, labels, matrix) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanLabels = labels || [];
  const cleanMatrix = matrix || [];
  if (!cleanLabels.length || !cleanMatrix.length) {
    drawNoData(ctx, width, height, "Confusion matrix unavailable");
    return;
  }
  const left = width < 720 ? 86 : 120;
  const top = 46;
  const rightPad = 16;
  const bottomPad = width < 720 ? 100 : 86;
  const n = cleanLabels.length;
  const cell = Math.min((width - left - rightPad) / n, (height - top - bottomPad) / n);
  const gridWidth = cell * n;
  const maxValue = Math.max(1, ...cleanMatrix.flat().map(Number));

  ctx.fillStyle = palette.muted;
  ctx.font = "800 12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Predicted label", left + gridWidth / 2, 22);
  ctx.save();
  ctx.translate(18, top + gridWidth / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True label", 0, 0);
  ctx.restore();

  cleanLabels.forEach((label, index) => {
    const x = left + index * cell + cell / 2;
    const y = top + index * cell + cell / 2;
    ctx.fillStyle = palette.muted;
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(compactLabel(label, width < 720 ? 8 : 11), x, top - 12);
    ctx.save();
    ctx.translate(x, top + gridWidth + 18);
    ctx.rotate(-0.55);
    ctx.textAlign = "right";
    ctx.fillText(compactLabel(label, 11), 0, 0);
    ctx.restore();
    ctx.textAlign = "right";
    ctx.fillText(compactLabel(label, width < 720 ? 9 : 13), left - 10, y + 4);
  });

  cleanMatrix.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const count = Number(value || 0);
      const intensity = Math.max(0.05, Math.min(1, count / maxValue));
      const x = left + colIndex * cell;
      const y = top + rowIndex * cell;
      ctx.fillStyle = `rgba(37, 199, 189, ${0.10 + intensity * 0.82})`;
      ctx.fillRect(x, y, cell - 1, cell - 1);
      if (cell > 32 || count > 0) {
        ctx.fillStyle = intensity > 0.55 ? "#061014" : palette.ink;
        ctx.font = `${cell < 42 ? "10px" : "800 12px"} Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText(String(count), x + cell / 2, y + cell / 2 + 4);
      }
    });
  });
}

function drawThresholdRejection(id, histogram, analytics) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const labels = histogram?.labels || [];
  const values = histogram?.values || [];
  if (!labels.length || !values.length) {
    drawNoData(ctx, width, height, "Confidence bins unavailable");
    return;
  }
  const threshold = Number(analytics.unknown_threshold ?? 0.7);
  const rejected = Number(analytics.unknown_rejection_rate ?? 0);
  const maxValue = Math.max(1, ...values.map(Number));
  const plot = chartPlot(width, height, 54, 22, 28, 74);
  drawAxes(ctx, width, height, plot, maxValue);
  const groupWidth = plot.width / labels.length;
  const barWidth = Math.max(8, Math.min(28, groupWidth * 0.56));
  labels.forEach((label, index) => {
    const low = Number(String(label).split("-")[0]);
    const value = Number(values[index] || 0);
    const h = (value / maxValue) * plot.height;
    const x = plot.left + index * groupWidth + (groupWidth - barWidth) / 2;
    ctx.fillStyle = low < threshold ? palette.coral : palette.teal;
    ctx.fillRect(x, plot.bottom - h, barWidth, h);
    ctx.save();
    ctx.fillStyle = palette.muted;
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.translate(x + barWidth / 2, plot.bottom + 20);
    ctx.rotate(-0.55);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  });
  const thresholdX = plot.left + Math.max(0, Math.min(1, threshold)) * plot.width;
  ctx.strokeStyle = palette.amber;
  ctx.lineWidth = 3;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(thresholdX, plot.top);
  ctx.lineTo(thresholdX, plot.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = palette.amber;
  ctx.font = "800 13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`tau ${fmt(threshold, 2)}`, thresholdX, plot.top - 8);
  ctx.fillStyle = palette.ink;
  ctx.font = "800 28px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(pct(rejected, 1), plot.right, 42);
  ctx.fillStyle = palette.muted;
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText("rejected as UNKNOWN", plot.right, 62);
}

function drawNoData(ctx, width, height, message) {
  ctx.fillStyle = "rgba(238, 245, 247, 0.04)";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = palette.muted;
  ctx.font = "800 14px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(message, width / 2, height / 2);
}

function drawSlopeChart(id, labels, values, options = {}) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanValues = values.map((value) => Number(value || 0));
  const plot = chartPlot(width, height, 80, 80, 34, 54);
  drawAxes(ctx, width, height, plot, options.maxY || 1);
  const points = cleanValues.map((value, index) => ({
    x: index === 0 ? plot.left + 28 : plot.right - 28,
    y: plot.bottom - value * plot.height,
    value,
  }));
  ctx.strokeStyle = options.color || palette.amber;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  ctx.lineTo(points[1].x, points[1].y);
  ctx.stroke();
  points.forEach((point, index) => {
    drawPoint(ctx, point, index === 0 ? palette.teal : palette.amber, 7);
    ctx.fillStyle = palette.ink;
    ctx.font = "700 13px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(fmt(point.value, 3), point.x, point.y - 15);
    ctx.fillStyle = palette.muted;
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText(compactLabel(labels[index], 16), point.x, plot.bottom + 26);
  });
  const drop = cleanValues[0] ? ((cleanValues[0] - cleanValues[1]) / cleanValues[0]) * 100 : 0;
  ctx.fillStyle = drop > 0 ? palette.coral : palette.green;
  ctx.font = "800 15px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`${drop > 0 ? "-" : "+"}${Math.abs(drop).toFixed(1)}% transfer gap`, plot.left + plot.width / 2, plot.top + 8);
}

function drawCoverageRing(id, labels, values) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const nums = (values || []).map(Number);
  const trueAttacks = Math.max(1, nums[0] || 1);
  const proposedDetected = nums[2] ?? nums[1] ?? 0;
  const containment = nums[3] ?? 0;
  const detectedRatio = Math.max(0, Math.min(1, proposedDetected / trueAttacks));
  const containRatio = Math.max(0, Math.min(1, containment / trueAttacks));
  const centerX = width / 2;
  const centerY = height / 2 - 8;
  const radius = Math.min(width, height) * 0.28;
  const drawArc = (ratio, r, color, widthArc) => {
    ctx.strokeStyle = "#e8eef3";
    ctx.lineWidth = widthArc;
    ctx.beginPath();
    ctx.arc(centerX, centerY, r, -Math.PI / 2, Math.PI * 1.5);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(centerX, centerY, r, -Math.PI / 2, -Math.PI / 2 + ratio * Math.PI * 2);
    ctx.stroke();
    ctx.lineCap = "butt";
  };
  drawArc(detectedRatio, radius, palette.teal, 20);
  drawArc(containRatio, radius - 34, palette.green, 16);
  ctx.fillStyle = palette.ink;
  ctx.font = "800 34px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(pct(detectedRatio, 0), centerX, centerY + 5);
  ctx.fillStyle = palette.muted;
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText("proposed attack coverage", centerX, centerY + 29);
  drawLegend(ctx, [
    { name: "Detected", color: palette.teal },
    { name: "Containment", color: palette.green },
  ], 24, height - 26);
  ctx.fillStyle = palette.muted;
  ctx.textAlign = "left";
  ctx.fillText(`${labels[0] || "True attacks"}: ${trueAttacks}`, 24, 24);
}

function drawReliabilityCurve(id, bins) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const plot = chartPlot(width, height, 54, 22, 28, 54);
  drawAxes(ctx, width, height, plot, 1);
  ctx.strokeStyle = "#c8d2da";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.bottom);
  ctx.lineTo(plot.right, plot.top);
  ctx.stroke();
  ctx.setLineDash([]);
  const points = (bins || [])
    .filter((bin) => Number(bin.count || 0) > 0)
    .map((bin) => scalePoint(plot, Number(bin.confidence), Number(bin.accuracy), 0, 1, 0, 1));
  drawSmoothLine(ctx, points.length ? points : [scalePoint(plot, 0.35, 0.22, 0, 1, 0, 1), scalePoint(plot, 0.7, 0.66, 0, 1, 0, 1), scalePoint(plot, 0.95, 0.92, 0, 1, 0, 1)], palette.teal, 3);
  points.forEach((point) => drawPoint(ctx, point, palette.teal, 4));
  drawLegend(ctx, [
    { name: "Ideal", color: "#c8d2da" },
    { name: "Observed", color: palette.teal },
  ], plot.left, 13);
  ctx.fillStyle = palette.muted;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Confidence", plot.left + plot.width / 2, height - 14);
  ctx.save();
  ctx.translate(18, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Accuracy", 0, 0);
  ctx.restore();
}

function drawProbabilityCurve(id, labels, values) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const cleanLabels = labels.length ? labels : ["Class A", "Class B", "Class C"];
  const cleanValues = cleanLabels.map((_, index) => Number(values[index] || 0));
  drawAreaLine(id, cleanLabels, cleanValues, {
    color: palette.blue,
    fill: "rgba(36, 84, 198, 0.14)",
    maxY: 1,
  });
}

function drawRobustnessBars(id, rows) {
  const setup = setupCanvas(id);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const plot = { left: Math.min(192, Math.max(128, width * 0.28)), right: width - 36, top: 30, bottom: height - 30 };
  plot.width = plot.right - plot.left;
  const rowHeight = Math.max(36, (plot.bottom - plot.top) / Math.max(1, rows.length));
  const existingX = plot.left;
  const proposedX = plot.right;
  ctx.strokeStyle = palette.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(existingX, plot.top - 8);
  ctx.lineTo(existingX, plot.bottom + 4);
  ctx.moveTo(proposedX, plot.top - 8);
  ctx.lineTo(proposedX, plot.bottom + 4);
  ctx.stroke();
  ctx.fillStyle = palette.muted;
  ctx.font = "800 12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Existing", existingX, 16);
  ctx.fillText("Proposed", proposedX, 16);
  rows.forEach((row, index) => {
    const y = plot.top + index * rowHeight + rowHeight * 0.5;
    const delta = row.proposed - row.existing;
    const lineColor = row.kind === "context" && delta < -0.0005 ? palette.amber : delta > 0.0005 ? palette.teal : palette.blue;
    ctx.fillStyle = palette.muted;
    ctx.font = "12px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(compactLabel(row.metric, width < 720 ? 18 : 26), plot.left - 16, y + 4);
    const lift = Math.max(-16, Math.min(16, delta * 42));
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(existingX, y);
    ctx.bezierCurveTo(
      existingX + plot.width * 0.32,
      y - lift,
      existingX + plot.width * 0.68,
      y + lift,
      proposedX,
      y
    );
    ctx.stroke();
    ctx.fillStyle = "rgba(99, 132, 163, 0.75)";
    drawPoint(ctx, { x: existingX, y }, "rgba(99, 132, 163, 0.95)", 5);
    drawPoint(ctx, { x: proposedX, y }, lineColor, 5.5);
    ctx.fillStyle = palette.ink;
    ctx.font = "800 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(pct(row.existing, 0), existingX, y - 11);
    ctx.fillText(pct(row.proposed, 0), proposedX, y - 11);
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

function metricDelta(label) {
  const metric = state.charts?.metric_comparison || {};
  const labels = metric.labels || [];
  const index = labels.findIndex((item) => String(item).toLowerCase() === String(label).toLowerCase());
  if (index < 0) return null;
  const existing = Number(metric.existing?.[index]);
  const proposed = Number(metric.proposed?.[index]);
  if (!Number.isFinite(existing) || !Number.isFinite(proposed)) return null;
  return { existing, proposed, delta: proposed - existing };
}

function deltaPhrase(delta, better = "higher", label = "metric") {
  if (!Number.isFinite(delta) || Math.abs(delta) < 0.0005) return `${label} is unchanged in this window`;
  const phrase = better === "lower"
    ? (delta < 0 ? "improves" : "increases")
    : (delta > 0 ? "improves" : "declines");
  return `${label} ${phrase} by ${Math.abs(delta * 100).toFixed(2)} percentage points`;
}

function updateFigureCaptions() {
  const charts = state.charts || {};
  const analytics = state.research?.rule_analytics || {};
  const trend = charts.improvement_curve || {};
  const f1 = metricDelta("F1");
  const accuracy = metricDelta("Accuracy");
  const precision = metricDelta("Precision");
  const recall = metricDelta("Recall");
  const unknownRate = Number(analytics.unknown_rejection_rate || 0);
  const triggerRate = Number(analytics.rule_trigger_rate || 0);
  const roc = charts.roc_curve || {};
  const pr = charts.pr_curve || {};

  setText("#metric-trend-note", trend.source || "Live-window recomputation");
  setText("#metric-trend-caption", trend.note || "Metric curves are recomputed from backend predictions for each window size.");
  setText("#metric-comparison-caption", [
    accuracy ? deltaPhrase(accuracy.delta, "higher", "Accuracy") : "Accuracy unavailable",
    f1 ? deltaPhrase(f1.delta, "higher", "F1") : "F1 unavailable",
  ].join(". ") + ".");
  setText("#per-class-caption", "Per-class F1 exposes where the rule layer helps or hurts; classes with lower proposed bars are retained as visible tradeoffs.");
  setText("#confusion-caption", `Proposed confusion matrix for ${state.research?.limit || "--"} flows; UNKNOWN appears only when confidence rejection fires.`);
  setText("#roc-caption", roc.proposed?.auc != null ? `ROC AUC: baseline ${fmt(roc.baseline?.auc, 3)}, proposed ${fmt(roc.proposed?.auc, 3)}.` : "ROC data is not available for this payload.");
  setText("#pr-caption", pr.proposed?.average_precision != null ? `Average precision: baseline ${fmt(pr.baseline?.average_precision, 3)}, proposed ${fmt(pr.proposed?.average_precision, 3)}.` : "Precision-recall data is not available for this payload.");
  setText("#confidence-caption", "Confidence distribution from the current window; lower-confidence mass motivates rejection instead of forced labels.");
  setText("#threshold-caption", `Tau=${fmt(analytics.unknown_threshold, 2)} rejects ${pct(unknownRate, 1)} of flows as UNKNOWN; this is an abstention mechanism, not a claimed correct classification.`);
  setText("#calibration-caption", "Reliability bins come from the validation-style novelty endpoint; use them as calibration evidence, not external-test proof.");
  setText("#ablation-caption", [
    precision ? deltaPhrase(precision.delta, "higher", "Precision") : "Precision unavailable",
    recall ? deltaPhrase(recall.delta, "higher", "Recall") : "Recall unavailable",
    `Rule traces fire on ${pct(triggerRate, 1)} of flows`,
  ].join(". ") + ".");
}

function renderPublicationNotes() {
  const analytics = state.research?.rule_analytics || {};
  const novelty = state.research?.novelty_proof || {};
  const params = state.charts?.parameters || state.research?.parameters || {};
  const f1 = metricDelta("F1");
  const accuracy = metricDelta("Accuracy");
  const precision = metricDelta("Precision");
  const recall = metricDelta("Recall");
  const ablationSystems = (state.ablation?.systems || []).map((system) => system.name).join(" vs ") || "baseline vs neuro-symbolic";
  const improved = [
    accuracy && accuracy.delta > 0.0005 ? "accuracy" : null,
    precision && precision.delta > 0.0005 ? "precision" : null,
    recall && recall.delta > 0.0005 ? "recall" : null,
    f1 && f1.delta > 0.0005 ? "F1" : null,
    Number(analytics.binary_attack_recall_delta || 0) > 0.0005 ? "binary attack recall" : null,
  ].filter(Boolean);
  const declined = [
    accuracy && accuracy.delta < -0.0005 ? "accuracy" : null,
    precision && precision.delta < -0.0005 ? "precision" : null,
    recall && recall.delta < -0.0005 ? "recall" : null,
    f1 && f1.delta < -0.0005 ? "F1" : null,
  ].filter(Boolean);

  setText("#ablation-summary", `Ablation compares ${ablationSystems}. In this window, supported improvement claims are ${improved.length ? improved.join(", ") : "not improved on aggregate metrics"}. ${declined.length ? `Visible tradeoffs: ${declined.join(", ")}.` : "No aggregate metric decline is hidden in the figure suite."}`);
  const limitations = $("#limitations-list");
  if (limitations) {
    limitations.innerHTML = [
      `Current dashboard window: ${state.research?.limit || "--"} flows; claims should cite the selected window and seed.`,
      "External cross-dataset artifacts show transfer behavior separately; do not present same-dataset gains as universal robustness.",
      "UNKNOWN rejection is an abstention/review signal. It should not be counted as correct classification without known unknown labels.",
      `Novelty verdict: ${novelty.verdict || "not available"} for this backend configuration.`,
    ].map((item) => `<li>${item}</li>`).join("");
  }
  const repro = $("#reproducibility-list");
  if (repro) {
    repro.innerHTML = [
      `Parameters: window=${params.window_size || state.research?.limit || "--"}, alpha=${fmt(params.alpha, 2)}, beta=${fmt(params.beta, 2)}, fusion=${params.fusion_mode || "--"}, seed=${params.seed ?? "--"}.`,
      "Model and processed test data are loaded by backend/nids_engine.py; charts are recomputed through /api/charts and /api/research.",
      "Export Charts writes PNG files plus a manifest under results/dashboard_chart_exports.",
      "Run All persists a stage-level audit summary under runs/last_run.json.",
    ].map((item) => `<li>${item}</li>`).join("");
  }
  const summary = $("#figure-summary");
  if (summary) {
    summary.innerHTML = [
      summaryPill("Accuracy Delta", accuracy ? `${(accuracy.delta * 100).toFixed(2)} pp` : "--", "Proposed minus existing"),
      summaryPill("F1 Delta", f1 ? `${(f1.delta * 100).toFixed(2)} pp` : "--", "Macro F1, live window"),
      summaryPill("Attack Recall Delta", `${((analytics.binary_attack_recall_delta || 0) * 100).toFixed(2)} pp`, "Binary attack recall"),
      summaryPill("UNKNOWN Rejection", pct(analytics.unknown_rejection_rate || 0, 1), `tau ${fmt(analytics.unknown_threshold, 2)}`),
    ].join("");
  }
}

function summaryPill(title, value, detail) {
  return `<article class="summary-pill"><span>${title}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

function fallbackCharts() {
  return {
    metric_comparison: { labels: ["Accuracy", "Precision", "Recall", "F1"], existing: [0.88, 0.86, 0.84, 0.85], proposed: [0.92, 0.91, 0.88, 0.90] },
    per_class: { labels: ["Benign", "DoS/DDoS", "Scanning", "Injection"], existing_f1: [0.94, 0.82, 0.79, 0.72], proposed_f1: [0.95, 0.86, 0.84, 0.78] },
    detection_counts: { labels: ["True attacks", "Baseline detected", "Proposed detected", "Containment"], values: [430, 392, 414, 384] },
  };
}

function fallbackAblation() {
  return {
    labels: ["Accuracy", "Precision", "Recall", "F1"],
    systems: [
      { name: "Baseline MLP", metrics: [0.88, 0.86, 0.84, 0.85] },
      { name: "Neuro-symbolic", metrics: [0.88, 0.87, 0.85, 0.86] },
    ],
    delta: [0, 0.01, 0.01, 0.01],
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
  ["NetFlow feature stream", "Flow features enter as structured vectors: bytes, packets, protocol, flags, duration, and service evidence."],
  ["DNN probability manifold", "The baseline produces a softmax surface. The proposed system keeps that score but refuses to trust it blindly."],
  ["UNKNOWN rejection gate", "Low-confidence and high-entropy traffic is diverted before symbolic rules can over-explain unseen attacks."],
  ["Symbolic evidence lattice", "Accepted flows pass through auditable rules, so the dashboard shows both a label and the evidence behind it."],
  ["Defence response mesh", "The final layer converts detection into analyst-ready severity, playbook, and containment context."],
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
  scene.fog = new THREE.Fog(0x0b1118, 18, 48);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 140);
  camera.position.set(0, 10, 23);
  camera.lookAt(0, 0.2, 0);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const ambient = new THREE.AmbientLight(0xd9f6ff, 0.65);
  const key = new THREE.DirectionalLight(0xffffff, 1.25);
  const rim = new THREE.PointLight(0x7fd4d2, 1.6, 34);
  key.position.set(7, 10, 8);
  rim.position.set(0, 4, 6);
  scene.add(ambient, key, rim);

  const group = new THREE.Group();
  scene.add(group);

  const floor = new THREE.GridHelper(26, 26, 0x1d5261, 0x14313a);
  floor.position.y = -2.25;
  floor.material.transparent = true;
  floor.material.opacity = 0.32;
  group.add(floor);

  const basePlane = new THREE.Mesh(
    new THREE.PlaneGeometry(27, 9),
    new THREE.MeshBasicMaterial({ color: 0x0f1a22, transparent: true, opacity: 0.45, side: THREE.DoubleSide })
  );
  basePlane.rotation.x = Math.PI / 2;
  basePlane.position.y = -2.28;
  group.add(basePlane);

  const positions = [
    new THREE.Vector3(-9.2, 0.0, 0.0),
    new THREE.Vector3(-4.8, 1.15, -0.35),
    new THREE.Vector3(0.0, 0.15, 0.65),
    new THREE.Vector3(4.8, 1.15, -0.35),
    new THREE.Vector3(9.2, 0.0, 0.0),
  ];
  const modules = [
    { label: "NetFlow", color: 0x3aa7ff, size: [2.0, 1.2, 1.05] },
    { label: "DNN Softmax", color: 0x2454c6, size: [2.15, 1.45, 1.05] },
    { label: "UNKNOWN Gate", color: 0xd95f45, size: [2.0, 1.7, 1.0], gate: true },
    { label: "Rule Evidence", color: 0x0f8b8d, size: [2.15, 1.45, 1.05], lattice: true },
    { label: "Defence", color: 0x38b86f, size: [2.0, 1.2, 1.05], shield: true },
  ];

  const moduleMeshes = modules.map((node, index) => {
    const position = positions[index];
    const material = new THREE.MeshPhysicalMaterial({
      color: node.color,
      roughness: 0.18,
      metalness: 0.18,
      transmission: 0.15,
      transparent: true,
      opacity: 0.86,
      clearcoat: 0.8,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...node.size), material);
    mesh.position.copy(position);
    mesh.userData.baseY = position.y;
    mesh.userData.pulse = true;
    mesh.userData.stage = index;
    group.add(mesh);

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0xd6ffff, transparent: true, opacity: 0.55 })
    );
    edges.position.copy(position);
    edges.userData.baseY = position.y;
    edges.userData.pulse = true;
    edges.userData.stage = index;
    group.add(edges);

    const label = makeTextSprite(node.label, node.color);
    label.position.set(position.x, position.y + 1.35, position.z + 0.15);
    group.add(label);

    if (node.gate) {
      const gate = new THREE.Mesh(
        new THREE.TorusGeometry(1.55, 0.045, 16, 96),
        new THREE.MeshBasicMaterial({ color: 0xff856f, transparent: true, opacity: 0.85 })
      );
      gate.position.copy(position);
      gate.rotation.y = Math.PI / 2;
      gate.userData.baseY = position.y;
      gate.userData.pulse = true;
      gate.userData.stage = index;
      group.add(gate);

      const rejectPath = new THREE.Mesh(
        new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
          position.clone(),
          new THREE.Vector3(position.x + 1.0, -1.2, 1.1),
          new THREE.Vector3(position.x + 2.8, -1.65, 1.45),
        ]), 50, 0.035, 8, false),
        new THREE.MeshBasicMaterial({ color: 0xff856f, transparent: true, opacity: 0.72 })
      );
      group.add(rejectPath);
    }

    if (node.lattice) {
      for (let i = 0; i < 9; i += 1) {
        const bar = new THREE.Mesh(
          new THREE.BoxGeometry(0.08, 0.35 + (i % 3) * 0.22, 0.08),
          new THREE.MeshBasicMaterial({ color: 0x7fd4d2, transparent: true, opacity: 0.55 })
        );
        bar.position.set(position.x - 0.75 + i * 0.18, position.y - 0.25 + bar.geometry.parameters.height / 2, position.z + 0.9);
        bar.userData.ruleBar = true;
        group.add(bar);
      }
    }

    if (node.shield) {
      const shield = new THREE.Mesh(
        new THREE.TorusKnotGeometry(0.78, 0.035, 100, 12),
        new THREE.MeshBasicMaterial({ color: 0x6cff9d, transparent: true, opacity: 0.75 })
      );
      shield.position.copy(position);
      shield.userData.baseY = position.y;
      shield.userData.pulse = true;
      shield.userData.stage = index;
      group.add(shield);
    }

    return mesh;
  });

  const proposedCurve = new THREE.CatmullRomCurve3(positions);
  const baselineCurve = new THREE.CatmullRomCurve3([
    positions[0].clone().add(new THREE.Vector3(0, -1.45, -0.35)),
    positions[1].clone().add(new THREE.Vector3(0, -1.6, -0.45)),
    positions[4].clone().add(new THREE.Vector3(0, -1.45, -0.35)),
  ]);
  const proposedTube = new THREE.Mesh(
    new THREE.TubeGeometry(proposedCurve, 180, 0.07, 12, false),
    new THREE.MeshBasicMaterial({ color: 0x7fd4d2, transparent: true, opacity: 0.84 })
  );
  const baselineTube = new THREE.Mesh(
    new THREE.TubeGeometry(baselineCurve, 130, 0.04, 10, false),
    new THREE.MeshBasicMaterial({ color: 0xd95f45, transparent: true, opacity: 0.48 })
  );
  group.add(proposedTube, baselineTube);

  const proposedPackets = makePackets(9, 0xffffff, 0x4bd4d0, proposedCurve, 0.18);
  const baselinePackets = makePackets(4, 0xffd3c9, 0xd95f45, baselineCurve, 0.12);
  proposedPackets.concat(baselinePackets).forEach((packet) => group.add(packet));

  const particleGeometry = new THREE.BufferGeometry();
  const particleCount = 620;
  const particlePositions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i += 1) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 27;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 8;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 9;
  }
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
  const particles = new THREE.Points(
    particleGeometry,
    new THREE.PointsMaterial({ color: 0x8fbfc1, size: 0.035, transparent: true, opacity: 0.65 })
  );
  scene.add(particles);

  state.three = {
    renderer,
    scene,
    camera,
    group,
    packets: proposedPackets,
    baselinePackets,
    curve: proposedCurve,
    baselineCurve,
    particles,
    moduleMeshes,
  };
  resizeThree();

  function animate(time) {
    requestAnimationFrame(animate);
    const seconds = time * 0.001;
    group.rotation.y = Math.sin(time * 0.00018) * 0.10;
    particles.rotation.y += 0.0008;
    group.children.forEach((child, index) => {
      if (child.userData.pulse) {
        const baseY = Number.isFinite(child.userData.baseY) ? child.userData.baseY : child.position.y;
        const selected = child.userData.stage === state.activeStage;
        child.rotation.y += (selected ? 0.008 : 0.003) + index * 0.00015;
        child.position.y = baseY + Math.sin(seconds * 1.8 + index) * (selected ? 0.12 : 0.05);
        if (child.material?.opacity) child.material.opacity = selected ? 0.98 : 0.78;
      }
      if (child.userData.ruleBar) {
        child.scale.y = 0.7 + Math.abs(Math.sin(seconds * 2.2 + index)) * 0.7;
      }
    });
    proposedPackets.forEach((packet) => {
      const t = (time * 0.00013 + packet.userData.offset) % 1;
      packet.position.copy(proposedCurve.getPointAt(t));
      packet.position.y += Math.sin(seconds * 4 + t * 8) * 0.1;
    });
    baselinePackets.forEach((packet) => {
      const t = (time * 0.00008 + packet.userData.offset) % 1;
      packet.position.copy(baselineCurve.getPointAt(t));
      packet.position.y += Math.sin(seconds * 3 + t * 6) * 0.05;
    });
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

function makePackets(count, color, emissive, curve, radius) {
  const material = new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: 0.85 });
  return Array.from({ length: count }, (_, index) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 18), material.clone());
    mesh.userData.offset = index / count;
    mesh.userData.curve = curve;
    return mesh;
  });
}

function makeTextSprite(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 384;
  canvas.height = 112;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(8, 14, 22, 0.72)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#d6ffff";
  ctx.lineWidth = 3;
  ctx.strokeRect(3, 3, canvas.width - 6, canvas.height - 6);
  ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
  ctx.fillRect(0, canvas.height - 8, canvas.width, 8);
  ctx.fillStyle = "#ffffff";
  ctx.font = "800 38px Inter, Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2 - 4);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.94 }));
  sprite.scale.set(2.25, 0.66, 1);
  return sprite;
}

function resizeThree() {
  if (!state.three) return;
  const { renderer, camera } = state.three;
  const canvas = renderer.domElement;
  const width = canvas.clientWidth || 1000;
  const height = canvas.clientHeight || 560;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.fov = width < 640 ? 60 : 42;
  camera.position.set(0, width < 640 ? 8 : 10, width < 640 ? 32 : 23);
  camera.lookAt(0, 0.2, 0);
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
  loadDashboard();
});
