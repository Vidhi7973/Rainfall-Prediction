"use strict";

// ── Constants ────────────────────────────────────────
const ARC_LENGTH = 172;

const MODEL_KEYS = {
  "SVM":           "svm",
  "Decision Tree": "dt",
  "Random Forest": "rf",
};

// ── Theme-aware model colors ──────────────────────────
function getModelColors() {
  const isDark = document.documentElement.dataset.theme !== "light";
  return {
    svm: isDark ? "#a855f7" : "#9333ea",
    dt:  isDark ? "#f97316" : "#c2410c",
    rf:  isDark ? "#3b82f6" : "#2563eb",
  };
}
let MODEL_COLORS = getModelColors();

// ── Theme-aware chart style ───────────────────────────
const CHART_DEFAULTS = {};
function refreshChartDefaults() {
  const isDark = document.documentElement.dataset.theme !== "light";
  CHART_DEFAULTS.gridColor = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)";
  CHART_DEFAULTS.tickColor = isDark ? "#64748b" : "#475569";
  CHART_DEFAULTS.ttBg = isDark ? "#0c1526" : "#ffffff";
  CHART_DEFAULTS.ttBorder = isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.12)";
  Chart.defaults.color = CHART_DEFAULTS.tickColor;
}
refreshChartDefaults();

// Caches for re-rendering on theme switch
let cachedDiag = null;
let cachedPred = null;

// ── Theme Management ─────────────────────────────────
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("rainsense-theme", theme);
  document.querySelectorAll(".theme-btn").forEach(btn =>
    btn.classList.toggle("active", btn.dataset.theme === theme)
  );
  refreshChartDefaults();
  MODEL_COLORS = getModelColors();
  // Rebuild all charts with new palette
  buildPerfCharts();
  if (cachedDiag) renderDiagnostics(cachedDiag);
  if (cachedPred) renderResults(cachedPred);
}

document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("rainsense-theme") || "dark";
  // Set initial theme before charts render
  document.documentElement.dataset.theme = saved;
  refreshChartDefaults();
  MODEL_COLORS = getModelColors();
  document.querySelectorAll(".theme-btn").forEach(btn =>
    btn.classList.toggle("active", btn.dataset.theme === saved)
  );
  document.getElementById("themeToggle")?.addEventListener("click", e => {
    const btn = e.target.closest(".theme-btn");
    if (btn) applyTheme(btn.dataset.theme);
  });
});


// Registry to destroy chart instances before recreating
const chartReg = {};

function destroyChart(id) {
  if (chartReg[id]) { chartReg[id].destroy(); delete chartReg[id]; }
}

function mkChart(id, config) {
  destroyChart(id);
  const el = document.getElementById(id);
  if (!el) return null;
  chartReg[id] = new Chart(el, config);
  return chartReg[id];
}

// ── Global Chart.js defaults ─────────────────────────
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.font.size = 11;

function getTooltipDefaults() {
  const isDark = document.documentElement.dataset.theme !== "light";
  return {
    backgroundColor: CHART_DEFAULTS.ttBg,
    borderColor: CHART_DEFAULTS.ttBorder,
    borderWidth: 1,
    padding: 10,
    titleColor: isDark ? "#e2e8f0" : "#0f172a",
    bodyColor: isDark ? "#94a3b8" : "#475569",
  };
}
// Alias for backward compat — refreshed each buildPerfCharts call
let tooltipDefaults = getTooltipDefaults();

// ══════════════════════════════════════════════════════
//  PERFORMANCE CHARTS (built on load)
// ══════════════════════════════════════════════════════
function buildPerfCharts() {
  tooltipDefaults = getTooltipDefaults();
  const M = window.METRICS;
  const names = Object.keys(M);
  const colors = names.map(n => MODEL_COLORS[MODEL_KEYS[n]]);
  const gc = CHART_DEFAULTS.gridColor;
  const tc = CHART_DEFAULTS.tickColor;

  // ── 1. Grouped bar ──────────────────────────────────
  mkChart("groupedChart", {
    type: "bar",
    data: {
      labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
      datasets: names.map((n, i) => ({
        label: n,
        data: [M[n].accuracy, M[n].precision, M[n].recall, M[n].f1_score],
        backgroundColor: colors[i] + "cc",
        borderColor: colors[i],
        borderWidth: 1.5,
        borderRadius: 5,
        borderSkipped: false,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, boxHeight: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%` } },
      },
      scales: {
        y: { min: 50, max: 100, ticks: { color: tc, callback: v => v + "%" }, grid: { color: gc } },
        x: { ticks: { color: tc }, grid: { display: false } },
      },
    },
  });

  // ── 2. Bubble (Accuracy vs F1) ──────────────────────
  mkChart("scatterChart", {
    type: "bubble",
    data: {
      datasets: names.map((n, i) => ({
        label: n,
        data: [{ x: M[n].accuracy, y: M[n].f1_score, r: 16 }],
        backgroundColor: colors[i] + "55",
        borderColor: colors[i],
        borderWidth: 2,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.dataset.label}  Acc: ${ctx.raw.x}%  F1: ${ctx.raw.y}%` } },
      },
      scales: {
        x: { min: 70, max: 100, title: { display: true, text: "Accuracy (%)", color: tc }, ticks: { color: tc, callback: v => v + "%" }, grid: { color: gc } },
        y: { min: 50, max: 100, title: { display: true, text: "F1-Score (%)", color: tc }, ticks: { color: tc, callback: v => v + "%" }, grid: { color: gc } },
      },
    },
  });

  // ── 3. Radar ────────────────────────────────────────
  mkChart("radarChart", {
    type: "radar",
    data: {
      labels: ["Accuracy", "Precision", "Recall", "F1-Score"],
      datasets: names.map((n, i) => ({
        label: n,
        data: [M[n].accuracy, M[n].precision, M[n].recall, M[n].f1_score],
        backgroundColor: colors[i] + "22",
        borderColor: colors[i],
        borderWidth: 2,
        pointBackgroundColor: colors[i],
        pointRadius: 4,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults },
      },
      scales: {
        r: {
          min: 50, max: 100,
          ticks: { color: tc, backdropColor: "transparent", stepSize: 10, callback: v => v + "%" },
          grid: { color: gc }, angleLines: { color: gc },
          pointLabels: { color: "#94a3b8", font: { size: 11, weight: "600" } },
        },
      },
    },
  });

  // ── 4. Accuracy rings ────────────────────────────────
  names.forEach(n => {
    const key = MODEL_KEYS[n];
    const color = MODEL_COLORS[key];
    const acc = M[n].accuracy;
    mkChart(`accRing-${key}`, {
      type: "doughnut",
      data: {
        datasets: [{
          data: [acc, 100 - acc],
          backgroundColor: [color, "rgba(255,255,255,0.04)"],
          borderColor: [color, "transparent"],
          borderWidth: 2,
        }],
      },
      options: {
        responsive: false, cutout: "72%",
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { duration: 1400, easing: "easeInOutQuart" },
      },
    });
  });

  // ── 5. Per-metric mini charts ─────────────────────
  const metricDefs = [
    { id: "metAccChart", key: "accuracy", label: "Accuracy" },
    { id: "metPrecChart", key: "precision", label: "Precision" },
    { id: "metRecChart", key: "recall", label: "Recall" },
    { id: "metF1Chart", key: "f1_score", label: "F1-Score" },
  ];
  metricDefs.forEach(({ id, key }) => {
    mkChart(id, {
      type: "bar",
      data: {
        labels: names.map(n => MODEL_KEYS[n].toUpperCase()),
        datasets: [{
          data: names.map(n => M[n][key]),
          backgroundColor: colors.map(c => c + "cc"),
          borderColor: colors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.raw}%` } } },
        scales: {
          y: { min: 50, max: 100, ticks: { color: tc, callback: v => v + "%", maxTicksLimit: 4 }, grid: { color: gc } },
          x: { ticks: { color: tc }, grid: { display: false } },
        },
      },
    });
  });

  // ── 6. Feature importance (async) ───────────────────
  fetch("/api/feature-importance")
    .then(r => r.json())
    .then(data => buildFeatureChart(data))
    .catch(() => { });
}

function buildFeatureChart(data) {
  const colors = data.importances.map(v => {
    if (v >= 15) return "#3b82f6";
    if (v >= 10) return "#6366f1";
    if (v >= 7) return "#a855f7";
    return "#64748b";
  });
  mkChart("featureChart", {
    type: "bar",
    data: {
      labels: data.features,
      datasets: [{
        label: "Importance (%)",
        data: data.importances,
        backgroundColor: colors.map(c => c + "cc"),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 5,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.raw}% importance` } },
      },
      scales: {
        x: { ticks: { color: CHART_DEFAULTS.tickColor, callback: v => v + "%" }, grid: { color: CHART_DEFAULTS.gridColor } },
        y: { ticks: { color: "#94a3b8", font: { weight: "600" } }, grid: { display: false } },
      },
    },
  });
}

document.addEventListener("DOMContentLoaded", buildPerfCharts);

// ══════════════════════════════════════════════════════
//  SVG GAUGE (semicircle)
// ══════════════════════════════════════════════════════
function setGauge(key, probability) {
  const fill = document.getElementById(`gauge-fill-${key}`);
  const probEl = document.getElementById(`prob-${key}`);
  if (!fill || !probEl) return;

  const offset = ARC_LENGTH * (1 - probability / 100);
  fill.style.strokeDashoffset = ARC_LENGTH;
  fill.getBoundingClientRect(); // force reflow
  requestAnimationFrame(() => { fill.style.strokeDashoffset = offset; });

  const dur = 1200, t0 = performance.now();
  (function tick(now) {
    const p = Math.min((now - t0) / dur, 1);
    const e = 1 - Math.pow(1 - p, 3);
    probEl.textContent = Math.round(probability * e) + "%";
    if (p < 1) requestAnimationFrame(tick);
  })(performance.now());
}

// ══════════════════════════════════════════════════════
//  RENDER PREDICTION RESULTS
// ══════════════════════════════════════════════════════
function renderResults(predictions) {
  cachedPred = predictions;
  tooltipDefaults = getTooltipDefaults();

  document.getElementById("idleState").style.display = "none";
  document.getElementById("resultContent").style.display = "flex";

  const entries = Object.entries(predictions);
  const rainVotes = entries.filter(([, v]) => v.prediction === "Yes").length;
  const consensus = rainVotes >= 2 ? "Yes" : "No";
  const avgProb = entries.reduce((s, [, v]) => s + v.probability, 0) / entries.length;

  // ── Consensus banner ──────────────────────────────
  const banner = document.getElementById("consensusBanner");
  const iconEl = document.getElementById("consensusIcon");
  const verdictEl = document.getElementById("consensusVerdict");
  const subEl = document.getElementById("consensusSub");
  const probEl = document.getElementById("consensusProb");

  if (consensus === "Yes") {
    banner.className = "consensus-banner rain-bg";
    iconEl.className = "consensus-icon rain-icon"; iconEl.textContent = "🌧️";
    verdictEl.className = "consensus-verdict rain"; verdictEl.textContent = "Rain Expected Tomorrow";
    subEl.textContent = `${rainVotes}/3 models predict rain`;
    probEl.className = "consensus-prob rain";
  } else {
    banner.className = "consensus-banner norain-bg";
    iconEl.className = "consensus-icon norain-icon"; iconEl.textContent = "☀️";
    verdictEl.className = "consensus-verdict norain"; verdictEl.textContent = "No Rain Tomorrow";
    subEl.textContent = `${3 - rainVotes}/3 models predict clear skies`;
    probEl.className = "consensus-prob norain";
  }
  animateNumber(probEl, avgProb, 0, 1200, v => Math.round(v) + "%");

  // ── Gauge arcs ────────────────────────────────────
  const probBarsHTML = [];

  entries.forEach(([name, v]) => {
    const key = MODEL_KEYS[name];
    const isRain = v.prediction === "Yes";
    const card = document.getElementById(`card-${key}`);

    card.classList.toggle("rain-card", isRain);
    card.classList.toggle("norain-card", !isRain);
    setGauge(key, v.probability);

    const vb = document.getElementById(`verdict-${key}`);
    vb.textContent = isRain ? "RAIN" : "NO RAIN";
    vb.className = `gauge-verdict ${isRain ? "rain" : "norain"}`;

    const color = MODEL_COLORS[key];
    probBarsHTML.push(`
      <div class="pc-row">
        <div class="pc-model" style="color:${color}">${name}</div>
        <div class="pc-bar-wrap">
          <div class="pc-bar-fill" data-pct="${v.probability}" style="background:${color};width:0%"></div>
        </div>
        <div class="pc-prob" style="color:${color}">${v.probability}%</div>
      </div>`);
  });

  document.getElementById("probBars").innerHTML = probBarsHTML.join("");
  requestAnimationFrame(() => {
    document.querySelectorAll(".pc-bar-fill").forEach(el => {
      requestAnimationFrame(() => { el.style.width = el.dataset.pct + "%"; });
    });
  });

  // ── Donut charts (rain vs no-rain per model) ──────
  entries.forEach(([name, v]) => {
    const key = MODEL_KEYS[name];
    const color = MODEL_COLORS[key];
    const rainPct = v.probability;
    const noRainPct = 100 - rainPct;

    mkChart(`donut-${key}`, {
      type: "doughnut",
      data: {
        datasets: [{
          data: [rainPct, noRainPct],
          backgroundColor: [color, "rgba(255,255,255,0.05)"],
          borderColor: [color, "transparent"],
          borderWidth: 2,
        }],
      },
      options: {
        responsive: false, cutout: "68%",
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { duration: 1200, easing: "easeInOutQuart" },
      },
    });
    document.getElementById(`dv-${key}`).textContent = Math.round(rainPct) + "%";
  });

  // ── Polar area chart ──────────────────────────────
  const names = entries.map(([n]) => n);
  const probs = entries.map(([, v]) => v.probability);
  const colors = entries.map(([n]) => MODEL_COLORS[MODEL_KEYS[n]]);

  mkChart("probPolarChart", {
    type: "polarArea",
    data: {
      labels: names,
      datasets: [{
        data: probs,
        backgroundColor: colors.map(c => c + "55"),
        borderColor: colors,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 10, boxWidth: 8, usePointStyle: true, font: { size: 10 } } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%` } },
      },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { color: CHART_DEFAULTS.tickColor, backdropColor: "transparent", count: 4, callback: v => v + "%" },
          grid: { color: CHART_DEFAULTS.gridColor },
        },
      },
      animation: { duration: 1000 },
    },
  });

  // ── Certainty meter ───────────────────────────────
  const certainty = Math.abs(avgProb - 50) * 2; // 0–100%
  const certFill = document.getElementById("certFill");
  const certMarker = document.getElementById("certMarker");
  const certPct = document.getElementById("certPct");
  const certDesc = document.getElementById("certDesc");

  const level = certainty < 30 ? "Low" : certainty < 60 ? "Moderate" : certainty < 85 ? "High" : "Very High";
  certDesc.textContent = `${level} certainty · avg probability ${avgProb.toFixed(1)}%`;

  requestAnimationFrame(() => {
    certFill.style.width = certainty + "%";
    certMarker.style.left = certainty + "%";
  });
  animateNumber(certPct, certainty, 0, 1200, v => Math.round(v) + "%");

  // ── Vote dots ─────────────────────────────────────
  entries.forEach(([name, v]) => {
    const key = MODEL_KEYS[name];
    const wrap = document.getElementById(`vote-${key}`);
    const isRain = v.prediction === "Yes";
    wrap.className = `vote-dot-wrap ${isRain ? "vote-rain" : "vote-norain"}`;
  });

  // ── Model reliability panel ───────────────────────
  // Shows real accuracy/AUC/F1 from Rainfall.csv test set so the user
  // knows whether to trust each model's prediction above.
  const reliabilityRow = document.getElementById("reliabilityRow");
  if (reliabilityRow && cachedDiag) {
    const meta = window.META;
    reliabilityRow.innerHTML = entries.map(([name, v]) => {
      const key   = MODEL_KEYS[name];
      const color = MODEL_COLORS[key];
      const d     = cachedDiag[name];
      const isRain = v.prediction === "Yes";

      // AUC quality grade
      const auc = d.auc_roc;
      let aucLabel, aucClass;
      if      (auc >= 0.90) { aucLabel = "Excellent"; aucClass = "rel-grade-good"; }
      else if (auc >= 0.80) { aucLabel = "Very Good"; aucClass = "rel-grade-good"; }
      else if (auc >= 0.70) { aucLabel = "Good";      aucClass = "rel-grade-ok";   }
      else if (auc >= 0.60) { aucLabel = "Fair";      aucClass = "rel-grade-warn"; }
      else                  { aucLabel = "Poor";       aucClass = "rel-grade-bad";  }

      // Confidence in this specific prediction
      const confPct = Math.abs(v.probability - 50) * 2;
      const confLabel = confPct >= 70 ? "High confidence"
                      : confPct >= 40 ? "Moderate confidence"
                      : "Low confidence";

      return `
        <div class="rel-card">
          <div class="rel-header" style="border-color:${color}">
            <span class="rel-model-name" style="color:${color}">${name}</span>
            <span class="rel-prediction ${isRain ? "rel-rain" : "rel-norain"}">
              ${isRain ? "🌧 Rain" : "☀ No Rain"}
            </span>
          </div>
          <div class="rel-body">
            <div class="rel-stat">
              <span class="rel-val" style="color:${color}">${d.accuracy}%</span>
              <span class="rel-lbl">Accuracy on<br/>${meta.test_samples} real test records</span>
            </div>
            <div class="rel-stat">
              <span class="rel-val" style="color:${color}">${auc}</span>
              <span class="rel-lbl">ROC-AUC<br/><span class="${aucClass}">${aucLabel}</span></span>
            </div>
            <div class="rel-stat">
              <span class="rel-val" style="color:${color}">${d.f1_score}%</span>
              <span class="rel-lbl">F1-Score<br/>(precision × recall)</span>
            </div>
          </div>
          <div class="rel-footer">
            <div class="rel-conf-bar-wrap">
              <div class="rel-conf-bar" style="width:${v.probability}%;background:${color}"></div>
            </div>
            <div class="rel-conf-label">${confLabel} · ${v.probability}% rain probability</div>
          </div>
        </div>`;
    }).join("");
  }
}

// ── Helper: animate a number counter ─────────────────
function animateNumber(el, target, from = 0, duration = 1200, format = v => Math.round(v) + "%") {
  const t0 = performance.now();
  (function tick(now) {
    const p = Math.min((now - t0) / duration, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = format(from + (target - from) * e);
    if (p < 1) requestAnimationFrame(tick);
  })(performance.now());
}

// ══════════════════════════════════════════════════════
//  DIAGNOSTICS
// ══════════════════════════════════════════════════════
async function loadDiagnostics() {
  try {
    const res = await fetch("/api/diagnostics");
    const diag = await res.json();
    renderDiagnostics(diag);
  } catch {
    document.getElementById("diagLoading").textContent = "Diagnostics unavailable.";
  }
}

function renderDiagnostics(diag) {
  cachedDiag = diag;
  tooltipDefaults = getTooltipDefaults();

  document.getElementById("diagLoading").style.display = "none";
  document.getElementById("diagResults").style.display = "block";

  const meta = diag["_meta"];
  const modelNames = Object.keys(diag).filter(k => k !== "_meta");
  const colors = modelNames.map(n => MODEL_COLORS[MODEL_KEYS[n]]);
  const gc = CHART_DEFAULTS.gridColor;
  const tc = CHART_DEFAULTS.tickColor;

  // ── Verdict ───────────────────────────────────────
  const aucs = modelNames.map(n => diag[n].auc_roc);
  const minAUC = Math.min(...aucs);
  const avgAUC = aucs.reduce((a, b) => a + b, 0) / aucs.length;
  const rfAcc = window.METRICS["Random Forest"].accuracy;
  const baseline = meta.baseline_accuracy;
  const lift = (rfAcc - baseline).toFixed(1);

  const banner = document.getElementById("verdictBanner");
  const iconWrap = document.getElementById("verdictIconWrap");
  const titleEl = document.getElementById("verdictTitle");
  const detEl = document.getElementById("verdictDetail");
  const statsEl = document.getElementById("verdictStats");

  let verdictClass, icon, title, detail;

  if (minAUC >= 0.78) {
    verdictClass = "v-good"; icon = "✅";
    title = "Genuine Learning Confirmed";
    detail = `All three models are learning real meteorological patterns — not guessing. `
      + `The lowest AUC across models is ${minAUC.toFixed(3)}, well above the 0.5 random baseline. `
      + `Random Forest outperforms the majority-class guesser by +${lift}% accuracy.`;
  } else if (minAUC >= 0.63) {
    verdictClass = "v-ok"; icon = "⚠️";
    title = "Models Show Meaningful Patterns (with Caveats)";
    detail = `Models are learning, but at least one has an AUC below 0.78, suggesting room for improvement. `
      + `This may be due to class imbalance or limited feature expressiveness. Review the confusion matrices below.`;
  } else {
    verdictClass = "v-warn"; icon = "🚨";
    title = "Warning: Possible Random Behaviour Detected";
    detail = `One or more models have AUC near 0.5, indicating they may be predicting randomly. `
      + `Check class balance, feature quality, and consider retraining with more data.`;
  }

  banner.className = `verdict-banner ${verdictClass}`;
  iconWrap.textContent = icon;
  titleEl.textContent = title;
  detEl.textContent = detail;

  statsEl.innerHTML = `
    <div class="vs-item"><span class="vs-val">${avgAUC.toFixed(3)}</span><span class="vs-lbl">Avg AUC</span></div>
    <div class="vs-item"><span class="vs-val">+${lift}%</span><span class="vs-lbl">Accuracy lift</span></div>
    <div class="vs-item"><span class="vs-val">${baseline}%</span><span class="vs-lbl">Baseline acc.</span></div>
  `;

  // ── Class balance pie ─────────────────────────────
  const cb = meta.class_balance;
  mkChart("classBalChart", {
    type: "doughnut",
    data: {
      labels: ["No Rain", "Rain"],
      datasets: [{
        data: [cb.no_rain_pct, cb.rain_pct],
        backgroundColor: ["rgba(16,185,129,.6)", "rgba(59,130,246,.6)"],
        borderColor: ["#10b981", "#3b82f6"],
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      plugins: {
        legend: { position: "bottom", labels: { padding: 12, boxWidth: 10, usePointStyle: true, font: { size: 11 } } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw}%` } },
      },
      animation: { duration: 1000 },
    },
  });

  document.getElementById("balNote").innerHTML =
    `<strong>${cb.rain_pct}%</strong> rain · <strong>${cb.no_rain_pct}%</strong> no-rain`
    + ` across ${cb.total_samples.toLocaleString()} samples. `
    + (Math.abs(cb.rain_pct - cb.no_rain_pct) > 20
      ? "⚠️ <em>Moderate class imbalance — models use class_weight='balanced' to compensate.</em>"
      : "✓ <em>Reasonably balanced — models won't be biased by class skew.</em>");

  // ── Baseline comparison bar ───────────────────────
  const accData = modelNames.map(n => window.METRICS[n].accuracy);
  mkChart("baselineChart", {
    type: "bar",
    data: {
      labels: [...modelNames.map(n => MODEL_KEYS[n].toUpperCase()), "Baseline"],
      datasets: [{
        label: "Accuracy (%)",
        data: [...accData, baseline],
        backgroundColor: [...colors.map(c => c + "cc"), "rgba(100,116,139,.4)"],
        borderColor: [...colors, "#64748b"],
        borderWidth: 1.5, borderRadius: 6, borderSkipped: false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.raw}%` } },
        annotation: {},
      },
      scales: {
        y: { min: 50, max: 100, ticks: { color: tc, callback: v => v + "%" }, grid: { color: gc } },
        x: { ticks: { color: tc }, grid: { display: false } },
      },
    },
  });

  document.getElementById("baselineNote").innerHTML =
    `A dumb guesser that always picks the majority class scores <strong>${baseline}%</strong>. `
    + `Random Forest beats it by <strong>+${lift}%</strong> — confirming the models add value beyond chance.`;

  // ── AUC bars ──────────────────────────────────────
  const aucWrap = document.getElementById("aucBarsWrap");
  function aucGrade(v) {
    if (v >= 0.9) return ["Excellent", "grade-excellent"];
    if (v >= 0.8) return ["Very Good", "grade-good"];
    if (v >= 0.7) return ["Good", "grade-good"];
    if (v >= 0.6) return ["Fair", "grade-fair"];
    return ["Poor — near random", "grade-poor"];
  }

  aucWrap.innerHTML = modelNames.map((n, i) => {
    const auc = diag[n].auc_roc;
    const [label, cls] = aucGrade(auc);
    const fillPct = auc * 100;          // track covers 0→1 mapped to 0→100%
    return `
      <div class="auc-bar-row">
        <div class="auc-bar-header">
          <span class="auc-bar-name" style="color:${colors[i]}">${n}</span>
          <div style="display:flex;align-items:center;gap:8px">
            <span class="auc-grade ${cls}">${label}</span>
            <span class="auc-bar-score" style="color:${colors[i]}">${auc}</span>
          </div>
        </div>
        <div class="auc-track">
          <div class="auc-fill" style="width:0%;background:${colors[i]}" data-w="${fillPct}"></div>
        </div>
      </div>`;
  }).join("");

  // Animate AUC fills
  requestAnimationFrame(() => {
    document.querySelectorAll(".auc-fill").forEach(el => {
      requestAnimationFrame(() => { el.style.width = el.dataset.w + "%"; });
    });
  });

  // ── ROC curves ────────────────────────────────────
  const rocDatasets = modelNames.map((n, i) => {
    const pts = diag[n].roc_curve;
    return {
      label: `${n}  (AUC ${diag[n].auc_roc})`,
      data: pts.map(p => ({ x: p.fpr, y: p.tpr })),
      borderColor: colors[i],
      backgroundColor: "transparent",
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.3,
    };
  });

  // Random diagonal reference line
  rocDatasets.push({
    label: "Random Classifier (AUC = 0.5)",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: "rgba(255,255,255,0.2)",
    backgroundColor: "transparent",
    borderWidth: 1.5,
    borderDash: [6, 4],
    pointRadius: 0,
  });

  mkChart("rocChart", {
    type: "scatter",
    data: { datasets: rocDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      showLine: true,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` FPR: ${ctx.raw.x}  TPR: ${ctx.raw.y}` } },
      },
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "False Positive Rate (FPR)", color: tc }, ticks: { color: tc }, grid: { color: gc } },
        y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate (TPR)", color: tc }, ticks: { color: tc }, grid: { color: gc } },
      },
    },
  });

  // ── Confusion matrices ────────────────────────────
  const cmRow = document.getElementById("cmRow");
  cmRow.innerHTML = modelNames.map((n, i) => {
    const [[tn, fp], [fn, tp]] = diag[n].confusion_matrix;
    const total = tn + fp + fn + tp;
    const p = v => (v / total * 100).toFixed(1);
    return `
      <div class="cm-card">
        <div class="cm-model-label" style="color:${colors[i]}">${n}</div>
        <div class="cm-table">
          <div class="cm-corner"></div>
          <div class="cm-col-hdr">Predicted<br/>No Rain</div>
          <div class="cm-col-hdr">Predicted<br/>Rain</div>
          <div class="cm-row-hdr">Actual No Rain</div>
          <div class="cm-cell cm-tn">
            <div class="cm-cell-num">${tn}</div>
            <div class="cm-cell-pct">${p(tn)}%</div>
            <div class="cm-cell-lbl">True Neg ✓</div>
          </div>
          <div class="cm-cell cm-fp">
            <div class="cm-cell-num">${fp}</div>
            <div class="cm-cell-pct">${p(fp)}%</div>
            <div class="cm-cell-lbl">False Alarm ✗</div>
          </div>
          <div class="cm-row-hdr">Actual Rain</div>
          <div class="cm-cell cm-fn">
            <div class="cm-cell-num">${fn}</div>
            <div class="cm-cell-pct">${p(fn)}%</div>
            <div class="cm-cell-lbl">Missed Rain ✗</div>
          </div>
          <div class="cm-cell cm-tp">
            <div class="cm-cell-num">${tp}</div>
            <div class="cm-cell-pct">${p(tp)}%</div>
            <div class="cm-cell-lbl">True Pos ✓</div>
          </div>
        </div>
        <div class="cm-summary">
          <span>Sensitivity <strong>${diag[n].sensitivity}%</strong></span>
          <span>Specificity <strong>${diag[n].specificity}%</strong></span>
        </div>
      </div>`;
  }).join("");

  // ── Cross-validation chart ─────────────────────────
  mkChart("cvChart", {
    type: "bar",
    data: {
      labels: modelNames,
      datasets: [
        {
          label: "CV AUC Mean",
          data: modelNames.map(n => diag[n].cv_auc_mean),
          backgroundColor: colors.map(c => c + "cc"),
          borderColor: colors,
          borderWidth: 1.5, borderRadius: 6, borderSkipped: false,
        },
        {
          label: "± Std Dev (stability)",
          data: modelNames.map(n => diag[n].cv_auc_std),
          backgroundColor: colors.map(c => c + "44"),
          borderColor: colors.map(c => c + "88"),
          borderWidth: 1, borderRadius: 4, borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: {
          ...tooltipDefaults, callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? ` ${ctx.dataset.label}: ${ctx.raw}`
              : ` Std Dev: ±${ctx.raw} (lower = more stable)`,
          }
        },
      },
      scales: {
        y: { min: 0, max: 1, ticks: { color: tc }, grid: { color: gc } },
        x: { ticks: { color: tc }, grid: { display: false } },
      },
    },
  });

  // ── Sensitivity / Specificity grouped bar ─────────
  mkChart("sensSpecChart", {
    type: "bar",
    data: {
      labels: modelNames,
      datasets: [
        {
          label: "Sensitivity — Catches real rain (True Positive Rate)",
          data: modelNames.map(n => diag[n].sensitivity),
          backgroundColor: colors.map(c => c + "cc"),
          borderColor: colors,
          borderWidth: 1.5, borderRadius: 6, borderSkipped: false,
        },
        {
          label: "Specificity — Avoids false alarms (True Negative Rate)",
          data: modelNames.map(n => diag[n].specificity),
          backgroundColor: colors.map(c => c + "55"),
          borderColor: colors.map(c => c + "aa"),
          borderWidth: 1.5, borderRadius: 4, borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` ${ctx.dataset.label.split("—")[0].trim()}: ${ctx.raw}%` } },
      },
      scales: {
        y: { min: 0, max: 100, ticks: { color: tc, callback: v => v + "%" }, grid: { color: gc } },
        x: { ticks: { color: tc }, grid: { display: false } },
      },
    },
  });

  // ── Extended Metrics Table ────────────────────────
  const metricGroups = [
    {
      label: "Core Classification",
      metrics: [
        { key: "accuracy", label: "Accuracy", unit: "%", higher: true },
        { key: "balanced_accuracy", label: "Balanced Accuracy", unit: "%", higher: true },
        { key: "precision", label: "Precision (PPV)", unit: "%", higher: true },
        { key: "sensitivity", label: "Sensitivity / Recall (TPR)", unit: "%", higher: true },
        { key: "f1_score", label: "F1-Score", unit: "%", higher: true },
        { key: "specificity", label: "Specificity (TNR)", unit: "%", higher: true },
      ],
    },
    {
      label: "Derived Error Rates",
      metrics: [
        { key: "npv", label: "NPV — Negative Predictive Value", unit: "%", higher: true },
        { key: "fpr", label: "FPR — False Positive Rate", unit: "%", higher: false },
        { key: "fnr", label: "FNR — False Negative Rate", unit: "%", higher: false },
        { key: "fdr", label: "FDR — False Discovery Rate", unit: "%", higher: false },
        { key: "for_rate", label: "FOR — False Omission Rate", unit: "%", higher: false },
        { key: "g_mean", label: "G-Mean  (√TPR × TNR)", unit: "", higher: true },
      ],
    },
    {
      label: "Composite Scores",
      metrics: [
        { key: "mcc", label: "MCC — Matthews Correlation Coef.", unit: "", higher: true },
        { key: "kappa", label: "Cohen's Kappa", unit: "", higher: true },
        { key: "auc_roc", label: "ROC-AUC", unit: "", higher: true },
        { key: "auc_prc", label: "Precision-Recall AUC (AUPRC)", unit: "", higher: true },
      ],
    },
    {
      label: "Calibration",
      metrics: [
        { key: "log_loss", label: "Log Loss", unit: "", higher: false },
        { key: "brier_score", label: "Brier Score", unit: "", higher: false },
      ],
    },
    {
      label: "5-Fold Cross-Validation",
      metrics: [
        { key: "cv_auc_mean", label: "CV AUC Mean", unit: "", higher: true },
        { key: "cv_acc_mean", label: "CV Accuracy Mean", unit: "%", higher: true },
        { key: "cv_f1_mean", label: "CV F1-Score Mean", unit: "%", higher: true },
      ],
    },
  ];

  const tableContainer = document.getElementById("extMetricsTable");
  if (tableContainer) {
    let tableHTML = `<table class="ext-metrics-table"><thead><tr>
      <th>Metric</th>
      ${modelNames.map((n, i) => `<th style="color:${colors[i]}">${MODEL_KEYS[n].toUpperCase()}</th>`).join("")}
    </tr></thead><tbody>`;

    metricGroups.forEach(group => {
      tableHTML += `<tr class="metric-group-row"><td colspan="${modelNames.length + 1}">${group.label}</td></tr>`;
      group.metrics.forEach(({ key, label, unit, higher }) => {
        const vals = modelNames.map(n => diag[n][key]);
        const best = higher ? Math.max(...vals) : Math.min(...vals);
        tableHTML += `<tr>
          <td class="metric-name">${label}</td>
          ${vals.map((v, i) => {
          const isBest = (v === best);
          const display = unit === "%" ? `${v}%` : v;
          const style = isBest ? "" : `color:${colors[i]}`;
          return `<td class="metric-val${isBest ? " metric-best" : ""}" style="${style}">${display}</td>`;
        }).join("")}
        </tr>`;
      });
    });

    tableHTML += `</tbody></table>`;
    tableContainer.innerHTML = tableHTML;
  }

  // ── Precision-Recall Curves ───────────────────────
  const rainBaseline = meta.class_balance.rain_pct / 100;
  const prDatasets = modelNames.map((n, i) => ({
    label: `${MODEL_KEYS[n].toUpperCase()}  (AUPRC ${diag[n].auc_prc})`,
    data: diag[n].pr_curve.map(p => ({ x: p.recall, y: p.precision })),
    borderColor: colors[i],
    backgroundColor: "transparent",
    borderWidth: 2.5,
    pointRadius: 0,
    tension: 0.2,
  }));
  prDatasets.push({
    label: `Random Baseline (${meta.class_balance.rain_pct}% rain)`,
    data: [{ x: 0, y: rainBaseline }, { x: 1, y: rainBaseline }],
    borderColor: "rgba(255,255,255,0.2)",
    backgroundColor: "transparent",
    borderWidth: 1.5,
    borderDash: [6, 4],
    pointRadius: 0,
  });

  mkChart("prChart", {
    type: "scatter",
    data: { datasets: prDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      showLine: true,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` Recall: ${ctx.raw.x}  Precision: ${ctx.raw.y}` } },
      },
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "Recall", color: tc }, ticks: { color: tc }, grid: { color: gc } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision", color: tc }, ticks: { color: tc }, grid: { color: gc } },
      },
    },
  });

  // ── Calibration Curves ────────────────────────────
  const calDatasets = modelNames.map((n, i) => ({
    label: MODEL_KEYS[n].toUpperCase(),
    data: diag[n].cal_curve.map(p => ({ x: p.mean_pred, y: p.frac_pos })),
    borderColor: colors[i],
    backgroundColor: colors[i] + "40",
    borderWidth: 2,
    pointRadius: 5,
    pointBackgroundColor: colors[i],
    showLine: true,
    tension: 0.3,
  }));
  calDatasets.push({
    label: "Perfect Calibration",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: "rgba(255,255,255,0.25)",
    backgroundColor: "transparent",
    borderWidth: 1.5,
    borderDash: [5, 5],
    pointRadius: 0,
  });

  mkChart("calChart", {
    type: "scatter",
    data: { datasets: calDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      showLine: true,
      plugins: {
        legend: { position: "bottom", labels: { padding: 14, boxWidth: 10, usePointStyle: true } },
        tooltip: { ...tooltipDefaults, callbacks: { label: ctx => ` Mean Pred: ${ctx.raw.x}  Actual Frac: ${ctx.raw.y}` } },
      },
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "Mean Predicted Probability", color: tc }, ticks: { color: tc }, grid: { color: gc } },
        y: { min: 0, max: 1, title: { display: true, text: "Fraction of Positives", color: tc }, ticks: { color: tc }, grid: { color: gc } },
      },
    },
  });

  // ── Feature-Target Correlation Chart ─────────────
  const featureStats = meta.feature_stats;
  const featNames = Object.keys(featureStats);
  const featCorrs = featNames.map(f => featureStats[f].correlation);
  const featBgColors = featCorrs.map(c => c >= 0 ? "rgba(59,130,246,.75)" : "rgba(249,115,22,.75)");
  const featBrColors = featCorrs.map(c => c >= 0 ? "#3b82f6" : "#f97316");
  const featLabels = featNames.map(f => featureStats[f].significant ? `${f} *` : f);

  mkChart("featCorrChart", {
    type: "bar",
    data: {
      labels: featLabels,
      datasets: [{
        label: "Correlation with RainTomorrow",
        data: featCorrs,
        backgroundColor: featBgColors,
        borderColor: featBrColors,
        borderWidth: 1.5, borderRadius: 5, borderSkipped: false,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          ...tooltipDefaults,
          callbacks: {
            label: ctx => {
              const fn = featNames[ctx.dataIndex];
              const fs = featureStats[fn];
              const dir = fs.direction_ok ? "✓ Direction correct" : "✗ Direction mismatch";
              const sig = fs.significant ? `p=${fs.p_value} (significant)` : `p=${fs.p_value} (not significant)`;
              return [` Corr: ${ctx.raw}`, ` ${sig}`, ` ${dir}`];
            },
          },
        },
      },
      scales: {
        x: {
          min: -1, max: 1,
          title: { display: true, text: "Point-Biserial Correlation", color: tc },
          ticks: { color: tc, callback: v => v.toFixed(1) },
          grid: { color: gc },
        },
        y: { ticks: { color: "#94a3b8", font: { weight: "600" } }, grid: { display: false } },
      },
    },
  });

  // ── Data Verification Panel ───────────────────────
  const verifyPanel = document.getElementById("dataVerifyPanel");
  const verifyHTML = featNames.map(fname => {
    const fs = featureStats[fname];
    let statusClass, badgeText, badgeClass;
    if (!fs.significant) {
      statusClass = "v-insig"; badgeText = "Not Significant"; badgeClass = "badge-insig";
    } else if (fs.direction_ok) {
      statusClass = "v-ok"; badgeText = "Direction ✓"; badgeClass = "badge-ok";
    } else {
      statusClass = "v-warn"; badgeText = "Direction ✗"; badgeClass = "badge-warn";
    }
    const corrSign = fs.correlation >= 0 ? "+" : "";
    const expLabel = fs.expected_dir === "+" ? "↑ → more rain" : "↓ → less rain";
    return `
      <div class="verify-card ${statusClass}">
        <div class="verify-feat-name">${fname}</div>
        <div class="verify-corr">${corrSign}${fs.correlation}</div>
        <div class="verify-direction">Expected: ${expLabel}</div>
        <div class="verify-range">μ = ${fs.mean}  ±${fs.std}</div>
        <div class="verify-range">Range: [${fs.min}, ${fs.max}]</div>
        <span class="verify-badge ${badgeClass}">${badgeText}</span>
      </div>`;
  }).join("");
  verifyPanel.innerHTML = `<div class="verify-grid">${verifyHTML}</div>`;
}

document.addEventListener("DOMContentLoaded", loadDiagnostics);

// ══════════════════════════════════════════════════════
//  FORM SUBMIT
// ══════════════════════════════════════════════════════
const form = document.getElementById("predictForm");
const btn = document.getElementById("predictBtn");

form.addEventListener("submit", async e => {
  e.preventDefault();
  btn.classList.add("loading");
  btn.disabled = true;

  const fd = new FormData(form);
  const body = {};
  for (const [k, v] of fd.entries()) body[k] = v;
  if (!body.RainToday) body.RainToday = "0";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.success) renderResults(data.predictions);
    else alert("Prediction error: " + (data.error || "unknown"));
  } catch {
    alert("Cannot reach server. Make sure Flask is running on port 5000.");
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
});
