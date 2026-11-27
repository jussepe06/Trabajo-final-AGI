/**
 * FinGuard AI — Consola React 18 de Detección de Fraude y Anomalías (Bloque 2)
 * Basado en el sistema de diseño SaaS (Zinc / DM Sans / JetBrains Mono)
 */

const { useState, useEffect, useRef, useMemo } = React;
const html = htm.bind(React.createElement);

const PEN = new Intl.NumberFormat("es-PE", {
  style: "currency",
  currency: "PEN",
  minimumFractionDigits: 2,
});

const PRESETS = {
  normal: {
    label: "Cliente Normal",
    desc: "Compra diurna en Lima desde dispositivo habitual",
    monto_pen: 85.5,
    hora_dia: 14,
    distancia_km: 3.2,
    intentos_previos_1h: 0,
    es_internacional: 0,
    dispositivo_nuevo: 0,
    ratio_monto_promedio: 1.05,
    antiguedad_cuenta_dias: 840,
  },
  attack: {
    label: "Ataque Nocturno",
    desc: "Vaciado internacional de madrugada en dispositivo nuevo",
    monto_pen: 3450.0,
    hora_dia: 2,
    distancia_km: 240.0,
    intentos_previos_1h: 5,
    es_internacional: 1,
    dispositivo_nuevo: 1,
    ratio_monto_promedio: 6.4,
    antiguedad_cuenta_dias: 35,
  },
  evasive: {
    label: "Fraude Evasivo (< S/ 1,500)",
    desc: "Evade el Modo Base (<1500 S/ y nacional), pero es capturado por ML",
    monto_pen: 1290.0,
    hora_dia: 3,
    distancia_km: 145.0,
    intentos_previos_1h: 4,
    es_internacional: 0,
    dispositivo_nuevo: 1,
    ratio_monto_promedio: 5.2,
    antiguedad_cuenta_dias: 60,
  },
};

function SparkBar({ value, colorClass = "bg-blue-500" }) {
  const pct = Math.max(2, Math.min(100, value * 100));
  return html`
    <div className="flex items-center justify-end gap-2">
      <span className="font-mono text-xs font-medium">${value.toFixed(4)}</span>
      <div className="w-14 h-1.5 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden hidden sm:block">
        <div className=${`h-full rounded-full ${colorClass}`} style=${{ width: `${pct}%` }}></div>
      </div>
    </div>
  `;
}

function SupervisedCharts({ models, darkMode }) {
  const metricsCanvasRef = useRef(null);
  const finCanvasRef = useRef(null);
  const metricsChartRef = useRef(null);
  const finChartRef = useRef(null);

  useEffect(() => {
    if (!models || !models.length) return;

    const shortLabels = models.map((m) =>
      m.model_name
        .replace(" (Regla Heurística Estática)", "")
        .replace(" (KNN)", "")
        .replace(" (MLP)", "")
    );
    const textColor = darkMode ? "#a1a1aa" : "#52525b";
    const gridColor = darkMode ? "rgba(39, 39, 42, 0.7)" : "rgba(228, 228, 231, 0.8)";

    if (metricsChartRef.current) metricsChartRef.current.destroy();
    metricsChartRef.current = new Chart(metricsCanvasRef.current.getContext("2d"), {
      type: "bar",
      data: {
        labels: shortLabels,
        datasets: [
          {
            label: "Precision",
            data: models.map((m) => m.metrics.precision),
            backgroundColor: "#3b82f6",
            borderRadius: 4,
          },
          {
            label: "Recall (Sensibilidad)",
            data: models.map((m) => m.metrics.recall),
            backgroundColor: "#10b981",
            borderRadius: 4,
          },
          {
            label: "F1-Score",
            data: models.map((m) => m.metrics.f1_score),
            backgroundColor: "#8b5cf6",
            borderRadius: 4,
          },
          {
            label: "ROC-AUC",
            data: models.map((m) => m.metrics.roc_auc),
            backgroundColor: "#f59e0b",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 1.05,
            ticks: { color: textColor, font: { family: "JetBrains Mono", size: 11 } },
            grid: { color: gridColor },
          },
          x: {
            ticks: { color: textColor, font: { family: "DM Sans", size: 11, weight: "600" } },
            grid: { display: false },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: textColor, boxWidth: 10, font: { family: "DM Sans", size: 11 } },
          },
        },
      },
    });

    const netBenefits = models.map((m) => m.financial_simulation.beneficio_neto_pen);
    const maxBen = Math.max(...netBenefits);
    const finColors = netBenefits.map((v, idx) =>
      idx === 0 ? "#52525b" : Math.abs(v - maxBen) < 1e-2 ? "#10b981" : "#2563eb"
    );

    if (finChartRef.current) finChartRef.current.destroy();
    finChartRef.current = new Chart(finCanvasRef.current.getContext("2d"), {
      type: "bar",
      data: {
        labels: shortLabels,
        datasets: [
          {
            label: "Beneficio Neto (S/ PEN)",
            data: netBenefits,
            backgroundColor: finColors,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            ticks: {
              color: textColor,
              font: { family: "JetBrains Mono", size: 11 },
              callback: (val) => `S/ ${Number(val).toLocaleString("es-PE")}`,
            },
            grid: { color: gridColor },
          },
          x: {
            ticks: { color: textColor, font: { family: "DM Sans", size: 11, weight: "600" } },
            grid: { display: false },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: textColor, boxWidth: 10, font: { family: "DM Sans", size: 11 } },
          },
        },
      },
    });

    return () => {
      if (metricsChartRef.current) metricsChartRef.current.destroy();
      if (finChartRef.current) finChartRef.current.destroy();
    };
  }, [models, darkMode]);

  return html`
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Desempeño Estadístico por Modelo (Test Set)
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Comparativa simultánea de Precision, Recall, F1-Score y ROC-AUC
            </p>
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400">
            n = 1,000 tx
          </span>
        </div>
        <div className="h-64">
          <canvas ref=${metricsCanvasRef}></canvas>
        </div>
      </div>

      <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Simulación Financiera: Beneficio Neto (S/)
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Pérdida Evitada (TP) − Pérdida Sufrida (FN) − Costo Fricción (FP × S/ 25)
            </p>
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
            Óptimo Económico
          </span>
        </div>
        <div className="h-64">
          <canvas ref=${finCanvasRef}></canvas>
        </div>
      </div>
    </div>
  `;
}

function PcaScatterChart({ clustering, darkMode }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!clustering || !clustering.projection_sample) return;
    const sample = clustering.projection_sample;
    const textColor = darkMode ? "#a1a1aa" : "#52525b";
    const gridColor = darkMode ? "rgba(39, 39, 42, 0.65)" : "rgba(228, 228, 231, 0.8)";

    const normalPts = sample
      .filter((p) => p.clase_real === 0 && p.dbscan_outlier === 0)
      .map((p) => ({ x: p.pc1, y: p.pc2 }));
    const legitOutliers = sample
      .filter((p) => p.clase_real === 0 && p.dbscan_outlier === 1)
      .map((p) => ({ x: p.pc1, y: p.pc2 }));
    const capturedFrauds = sample
      .filter((p) => p.clase_real === 1 && p.dbscan_outlier === 1)
      .map((p) => ({ x: p.pc1, y: p.pc2 }));

    if (chartRef.current) chartRef.current.destroy();
    chartRef.current = new Chart(canvasRef.current.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Legítima Normal (Cluster Denso)",
            data: normalPts,
            backgroundColor: "rgba(113, 113, 122, 0.45)",
            pointRadius: 3,
          },
          {
            label: "Legítima Atípica (Outlier)",
            data: legitOutliers,
            backgroundColor: "rgba(245, 158, 11, 0.8)",
            pointRadius: 4,
          },
          {
            label: "Fraude Capturado (Outlier DBSCAN)",
            data: capturedFrauds,
            backgroundColor: "rgba(244, 63, 94, 0.92)",
            pointRadius: 4.5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Componente Principal 1 (PC1)", color: textColor, font: { size: 11 } },
            ticks: { color: textColor, font: { family: "JetBrains Mono", size: 10 } },
            grid: { color: gridColor },
          },
          y: {
            title: { display: true, text: "Componente Principal 2 (PC2)", color: textColor, font: { size: 11 } },
            ticks: { color: textColor, font: { family: "JetBrains Mono", size: 10 } },
            grid: { color: gridColor },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: { color: textColor, boxWidth: 10, font: { family: "DM Sans", size: 11 } },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [clustering, darkMode]);

  return html`<div className="h-72"><canvas ref=${canvasRef}></canvas></div>`;
}


/** Panel Bloque 1 — comparación verificable de estrategias con la misma muestra de test. */
function BlockOnePanel({ agentData, predictionData, error }) {
  const rows = agentData?.agents || [];
  const agentKeys = [
    ["modo_base", "Regla estática"],
    ["agente_modelo", "Basado en modelo"],
    ["agente_utilidad", "Basado en utilidad"],
  ];
  const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
  const time = (value) => `${Number(value || 0).toFixed(3)} ms`;

  return html`
    <section className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 sm:p-6 shadow-sm space-y-5">
      <div className="border-b border-zinc-200 dark:border-zinc-800 pb-4">
        <span className="text-xs font-semibold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
          Bloque 1 · Arquitecturas de agentes inteligentes
        </span>
        <h2 className="text-xl font-bold tracking-tight mt-1">¿Qué agente toma mejores decisiones?</h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
          Comparación de la regla estática, el agente basado en modelo y el agente basado en utilidad.
          Las tres estrategias se evalúan sobre el mismo conjunto de transacciones sintéticas.
        </p>
      </div>

      ${rows.length ? html`
        <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-zinc-100 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-300">
              <tr>
                ${["Estrategia", "Accuracy", "Recall", "F1", "FP", "FN", "Beneficio neto", "Predicción"].map((header) => html`
                  <th key=${header} className="px-3 py-3 whitespace-nowrap font-semibold">${header}</th>
                `)}
              </tr>
            </thead>
            <tbody>
              ${rows.map((agent) => html`
                <tr key=${agent.agent_key} className="border-t border-zinc-200 dark:border-zinc-800">
                  <td className="px-3 py-3 font-semibold">${agent.agent_name}</td>
                  <td className="px-3 py-3 font-mono">${percent(agent.metrics.accuracy)}</td>
                  <td className="px-3 py-3 font-mono">${percent(agent.metrics.recall)}</td>
                  <td className="px-3 py-3 font-mono text-blue-600 dark:text-blue-400 font-bold">${percent(agent.metrics.f1_score)}</td>
                  <td className="px-3 py-3 font-mono">${agent.confusion_matrix.fp}</td>
                  <td className="px-3 py-3 font-mono">${agent.confusion_matrix.fn}</td>
                  <td className="px-3 py-3 font-mono text-emerald-600 dark:text-emerald-400">${PEN.format(agent.financial_simulation.beneficio_neto_pen)}</td>
                  <td className="px-3 py-3 font-mono">${time(agent.prediction_time_ms)}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Muestra: ${agentData.test_rows} operaciones de prueba (${agentData.test_fraud_count} fraudes).
          Tiempos medidos por llamada, no promedios de varias corridas. Costo FP = S/ 25 (supuesto académico).
          El agente de utilidad utiliza un puntaje heurístico no calibrado como probabilidad.
        </p>
      ` : html`
        <div className="p-4 rounded-lg border border-amber-500/20 bg-amber-500/5 text-sm text-amber-700 dark:text-amber-300">
          ${error || "No hay métricas del Bloque 1 disponibles todavía."}
          Para activarlas, integra primero los PR de agents.py, ml_engine.py y main.py en main.
        </div>
      `}

      <div>
        <h3 className="text-sm font-semibold mb-2">Decisiones para la transacción del simulador</h3>
        ${predictionData ? html`
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            ${agentKeys.map(([key, label]) => {
              const decision = predictionData[key];
              if (!decision) return null;
              const blocked = decision.decision === "BLOQUEAR";
              return html`
                <div key=${key} className=${`rounded-lg border p-4 ${blocked
                  ? "border-rose-500/30 bg-rose-500/5"
                  : "border-emerald-500/30 bg-emerald-500/5"}`}>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">${label}</p>
                  <p className=${`text-base font-bold mt-1 ${blocked ? "text-rose-500" : "text-emerald-500"}`}>
                    ${decision.decision}
                  </p>
                  ${typeof decision.alert_count === "number" ? html`
                    <p className="text-xs text-zinc-500 mt-1">Alertas del estado: ${decision.alert_count}</p>
                  ` : null}
                  ${typeof decision.heuristic_risk_score === "number" ? html`
                    <p className="text-xs text-zinc-500 mt-1">Puntaje heurístico: ${decision.heuristic_risk_score.toFixed(3)}</p>
                  ` : null}
                </div>
              `;
            })}
          </div>
        ` : html`
          <p className="text-xs text-zinc-500">Selecciona un perfil en el simulador o ejecuta una predicción para ver los tres veredictos.</p>
        `}
      </div>
    </section>
  `;
}

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState("all");
  const [threshold, setThreshold] = useState(0.5);
  const [metricsData, setMetricsData] = useState(null);
  const [clusteringData, setClusteringData] = useState(null);
  const [agentsData, setAgentsData] = useState(null);
  const [agentPredictionData, setAgentPredictionData] = useState(null);
  const [agentLoadError, setAgentLoadError] = useState("");
  const [selectedModelKey, setSelectedModelKey] = useState("logistic_regression");

  // Estado del simulador en vivo (Panel 2)
  const [txForm, setTxForm] = useState(PRESETS.attack);
  const [activePreset, setActivePreset] = useState("attack");
  const [predictionResult, setPredictionResult] = useState(null);

  // Estado de re-entrenamiento (POST /api/retrain)
  const [seed, setSeed] = useState(42);
  const [fraudRate, setFraudRate] = useState(0.03);
  const [isRetraining, setIsRetraining] = useState(false);

  // Sincronizar clase dark en <html>
  useEffect(() => {
    const rootEl = document.documentElement;
    if (darkMode) rootEl.classList.add("dark");
    else rootEl.classList.remove("dark");
  }, [darkMode]);

  // Carga independiente del Bloque 1. Si todavía no se fusionaron sus PR,
  // el resto del dashboard continúa funcionando con normalidad.
  async function loadAgents() {
    try {
      const response = await fetch("/api/agents");
      if (!response.ok) throw new Error(`API Bloque 1: HTTP ${response.status}`);
      setAgentsData(await response.json());
      setAgentLoadError("");
    } catch (error) {
      setAgentLoadError("API del Bloque 1 no disponible todavía.");
      console.warn("Agentes pendientes de integración:", error);
    }
  }

  useEffect(() => {
    loadAgents();
  }, []);

  // Carga inicial de /api/clustering
  useEffect(() => {
    fetch("/api/clustering")
      .then((r) => r.json())
      .then(setClusteringData)
      .catch(console.error);
  }, []);

  // Carga reactiva de /api/metrics y /api/predict cuando cambia el threshold
  useEffect(() => {
    const timer = setTimeout(() => {
      fetch(`/api/metrics?threshold=${threshold.toFixed(2)}`)
        .then((r) => r.json())
        .then(setMetricsData)
        .catch(console.error);

      runPrediction(txForm, threshold);
    }, 45);
    return () => clearTimeout(timer);
  }, [threshold]);

  async function runPrediction(formValues, th = threshold) {
    const payload = {
      monto_pen: Number(formValues.monto_pen),
      hora_dia: Number(formValues.hora_dia),
      distancia_km: Number(formValues.distancia_km),
      intentos_previos_1h: Number(formValues.intentos_previos_1h),
      es_internacional: Number(formValues.es_internacional),
      dispositivo_nuevo: Number(formValues.dispositivo_nuevo),
      ratio_monto_promedio: Number(formValues.ratio_monto_promedio),
      antiguedad_cuenta_dias: Number(formValues.antiguedad_cuenta_dias),
      threshold: Number(th),
    };
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const result = await res.json();
      setPredictionResult(result);
      if (result.block1_agent_predictions) {
        setAgentPredictionData(result.block1_agent_predictions);
      }
    }
    // Endpoint agregado en PR #4; un 404 no debe romper el simulador original.
    try {
      const agentsResponse = await fetch("/api/agents/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (agentsResponse.ok) setAgentPredictionData(await agentsResponse.json());
    } catch (error) {
      console.warn("Predicción del Bloque 1 no disponible:", error);
    }
  }

  function handlePresetSelect(key) {
    const preset = PRESETS[key];
    setActivePreset(key);
    setTxForm(preset);
    runPrediction(preset, threshold);
  }

  function handleFieldChange(field, value) {
    const next = { ...txForm, [field]: value };
    setActivePreset("custom");
    setTxForm(next);
  }

  async function handleRetrain() {
    setIsRetraining(true);
    try {
      const res = await fetch("/api/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: Number(seed),
          fraud_rate: Number(fraudRate),
          threshold: Number(threshold),
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setMetricsData(data);
        setClusteringData(data.unsupervised_clustering);
        if (data.block1_agents) setAgentsData(data.block1_agents);
        await loadAgents();
        await runPrediction(txForm, threshold);
      }
    } finally {
      setIsRetraining(false);
    }
  }

  const selectedModelDetail = useMemo(() => {
    if (!metricsData || !metricsData.supervised_models) return null;
    return (
      metricsData.supervised_models.find((m) => m.model_key === selectedModelKey) ||
      metricsData.supervised_models[0]
    );
  }, [metricsData, selectedModelKey]);

  if (!metricsData) {
    return html`
      <div className="min-h-screen flex items-center justify-center bg-zinc-50 dark:bg-[#09090b]">
        <div className="flex items-center gap-3 text-sm text-zinc-400 font-mono">
          <div className="h-4 w-4 rounded-full border-2 border-emerald-500 border-t-transparent animate-spin"></div>
          Inicializando motor neuronal y cargando métricas de FinGuard AI...
        </div>
      </div>
    `;
  }

  const models = metricsData.supervised_models;
  const maxF1 = Math.max(...models.map((m) => m.metrics.f1_score));
  const maxFin = Math.max(...models.map((m) => m.financial_simulation.beneficio_neto_pen));
  const baseFin = models[0].financial_simulation.beneficio_neto_pen;
  const finDelta = maxFin - baseFin;

  return html`
    <div className="min-h-screen flex flex-col">
      <!-- Top Navigation Bar (Minimal SaaS Chrome) -->
      <header className="sticky top-0 z-30 border-b border-zinc-200 dark:border-zinc-800/90 bg-white/85 dark:bg-[#09090b]/85 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm shadow-sm shadow-blue-500/30">
              FG
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-zinc-950 dark:text-zinc-50">
                  FinGuard AI
                </h1>
                <span className="hidden sm:inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  Bloque 2 · Activo
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 hidden sm:block">
                Detección de Fraude Financiero &amp; Auditoría de Anomalías Transaccionales
              </p>
            </div>
          </div>

          <!-- Segmented View Switcher -->
          <nav className="hidden xl:flex items-center bg-zinc-100 dark:bg-[#0c0c0f] p-1 rounded-lg border border-zinc-200 dark:border-zinc-800 text-xs font-medium">
            ${[
              { id: "all", label: "Vista Integrada" },
              { id: "agents", label: "Agentes · Bloque 1" },
              { id: "panel1", label: "1. Umbral & Tabla Comparativa" },
              { id: "panel2", label: "2. Simulador Online" },
              { id: "panel3", label: "3. Clustering Anomalías" },
            ].map(
              (tab) => html`
                <button
                  key=${tab.id}
                  onClick=${() => setActiveTab(tab.id)}
                  className=${`px-3 py-1.5 rounded-md transition-all ${
                    activeTab === tab.id
                      ? "bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm font-semibold"
                      : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200"
                  }`}
                >
                  ${tab.label}
                </button>
              `
            )}
          </nav>

          <!-- Controles de Re-entrenamiento (POST /api/retrain) + Theme Toggle -->
          <div className="flex items-center gap-2">
            <div className="hidden md:flex items-center gap-2 bg-zinc-100 dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-2.5 py-1 text-xs">
              <span className="text-zinc-500">Seed:</span>
              <input
                type="number"
                value=${seed}
                onChange=${(e) => setSeed(e.target.value)}
                className="w-12 bg-transparent font-mono font-semibold text-zinc-900 dark:text-zinc-100 focus:outline-none"
              />
              <span className="text-zinc-300 dark:text-zinc-700">|</span>
              <span className="text-zinc-500">Fraude:</span>
              <select
                value=${fraudRate}
                onChange=${(e) => setFraudRate(parseFloat(e.target.value))}
                className="bg-transparent font-mono font-semibold text-zinc-900 dark:text-zinc-100 focus:outline-none"
              >
                <option value="0.02" className="bg-zinc-900 text-white">2%</option>
                <option value="0.03" className="bg-zinc-900 text-white">3%</option>
                <option value="0.05" className="bg-zinc-900 text-white">5%</option>
                <option value="0.08" className="bg-zinc-900 text-white">8%</option>
              </select>
              <button
                onClick=${handleRetrain}
                disabled=${isRetraining}
                className="ml-1 px-2.5 py-1 rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium transition-colors"
              >
                ${isRetraining ? "Entrenando..." : "Re-entrenar"}
              </button>
            </div>

            <button
              onClick=${() => setDarkMode(!darkMode)}
              title="Cambiar tema Claro / Oscuro"
              className="h-9 w-9 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#0c0c0f] hover:bg-zinc-100 dark:hover:bg-zinc-800 flex items-center justify-center text-sm transition-colors"
            >
              ${darkMode ? "☀️" : "🌙"}
            </button>
          </div>
        </div>
      </header>

      <!-- Main Dashboard Container -->
      <main className="flex-1 max-w-[1600px] w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        <!-- Fila de 4 KPI Cards (Estilo SaaS shadcn) -->
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Dataset Transaccional (80/20)
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
                Seed ${seed}
              </span>
            </div>
            <div className="text-2xl font-bold font-mono mt-2">
              ${metricsData.dataset_summary.total_rows.toLocaleString("es-PE")} tx
            </div>
            <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400 mt-2">
              <span>Legítimas: ${metricsData.dataset_summary.legit_count}</span>
              <span className="text-rose-500 font-mono font-medium">
                Fraudes: ${metricsData.dataset_summary.fraud_count} (${(
                  metricsData.dataset_summary.fraud_ratio * 100
                ).toFixed(1)}%)
              </span>
            </div>
          </div>

          <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Ganador en Beneficio Neto (S/)
              </span>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400">
                ↑ Óptimo Financiero
              </span>
            </div>
            <div className="text-xl font-bold text-emerald-600 dark:text-emerald-400 mt-2 truncate">
              ${metricsData.recommendations.best_financial_model}
            </div>
            <div className="flex items-center justify-between text-xs mt-2">
              <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100">
                ${PEN.format(metricsData.recommendations.best_financial_net_benefit_pen)}
              </span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400">
                +${PEN.format(Math.max(0, finDelta))} vs Base
              </span>
            </div>
          </div>

          <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Ganador en F1-Score (Balance)
              </span>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                ★ Top Clasificador
              </span>
            </div>
            <div className="text-xl font-bold text-blue-600 dark:text-blue-400 mt-2 truncate">
              ${metricsData.recommendations.best_f1_model}
            </div>
            <div className="flex items-center justify-between text-xs mt-2">
              <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100">
                F1 = ${metricsData.recommendations.best_f1_score.toFixed(4)}
              </span>
              <span className="text-zinc-500 font-mono">Umbral τ = ${threshold.toFixed(2)}</span>
            </div>
          </div>

          <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Matriz de Costo Bancario
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">
                Soles (PEN)
              </span>
            </div>
            <div className="text-base font-bold mt-2 text-zinc-900 dark:text-zinc-100">
              TP: +100% · FN: -100%
            </div>
            <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 flex justify-between">
              <span>Falso Bloqueo (FP):</span>
              <span className="font-mono font-semibold text-amber-500">-S/ 25.00 / tx</span>
            </div>
          </div>
        </section>

        <!-- =============================================================== -->
        <!-- BLOQUE 1: COMPARACIÓN DE ARQUITECTURAS DE AGENTES                -->
        <!-- =============================================================== -->
        ${(activeTab === "all" || activeTab === "agents") && html`
          <${BlockOnePanel}
            agentData=${agentsData || metricsData.block1_agents}
            predictionData=${agentPredictionData || predictionResult?.block1_agent_predictions}
            error=${agentLoadError}
          />
        `}

        <!-- =============================================================== -->
        <!-- PANEL 1: CONTROL DE UMBRAL Y TABLA COMPARATIVA EN VIVO          -->
        <!-- =============================================================== -->
        ${(activeTab === "all" || activeTab === "panel1") &&
        html`
          <section className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 sm:p-6 shadow-sm space-y-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-5">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
                    Panel 1 · Aprendizaje Supervisado (k=5 CV + Test Set)
                  </span>
                </div>
                <h2 className="text-xl font-bold tracking-tight mt-0.5">
                  Control de Umbral de Probabilidad y Tabla Comparativa en Vivo
                </h2>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                  Desliza el umbral para recalcular al instante Precision, Recall, F1-Score, Matriz de Confusión y Beneficio Neto (S/). Haz clic en una fila para inspeccionar su desglose financiero.
                </p>
              </div>

              <!-- Control Interactivo de Threshold -->
              <div className="bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-xl p-3.5 min-w-[300px] sm:min-w-[380px]">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300">
                    Umbral de Corte Probabilístico (τ):
                  </span>
                  <div className="flex items-center gap-1.5">
                    ${[0.25, 0.5, 0.75].map(
                      (val) => html`
                        <button
                          key=${val}
                          onClick=${() => setThreshold(val)}
                          className=${`text-[11px] font-mono px-2 py-0.5 rounded border transition ${
                            Math.abs(threshold - val) < 0.005
                              ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 font-bold"
                              : "bg-zinc-200/60 dark:bg-zinc-800 text-zinc-500 border-transparent hover:text-zinc-200"
                          }`}
                        >
                          ${val.toFixed(2)}
                        </button>
                      `
                    )}
                    <span className="ml-1 px-2.5 py-0.5 rounded-md bg-blue-600 text-white font-mono font-bold text-xs">
                      ${threshold.toFixed(2)}
                    </span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value=${threshold}
                  onChange=${(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-[11px] text-zinc-400 mt-1 font-mono">
                  <span>0.05 (Máx Recall)</span>
                  <span>0.50 (Balance)</span>
                  <span>0.95 (Máx Precision)</span>
                </div>
              </div>
            </div>

            <!-- Tabla Comparativa Principal + Panel Lateral de Inspección -->
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
              <div className="xl:col-span-8 overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
                <table className="w-full text-left border-collapse text-xs sm:text-sm">
                  <thead>
                    <tr className="bg-zinc-100 dark:bg-zinc-900/90 text-zinc-500 dark:text-zinc-400 border-b border-zinc-200 dark:border-zinc-800 text-[11px] uppercase tracking-wider">
                      <th className="py-3 px-3.5">Modelo / Técnica</th>
                      <th className="py-3 px-2.5 text-right">Precision</th>
                      <th className="py-3 px-2.5 text-right">Recall</th>
                      <th className="py-3 px-2.5 text-right">F1-Score</th>
                      <th className="py-3 px-2.5 text-right">ROC-AUC</th>
                      <th className="py-3 px-3 text-center">Matriz (TN / FP / FN / TP)</th>
                      <th className="py-3 px-3.5 text-right">Beneficio Neto (S/)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800/80">
                    ${models.map((item) => {
                      const m = item.metrics;
                      const cm = item.confusion_matrix;
                      const fin = item.financial_simulation;
                      const isBestF1 = Math.abs(m.f1_score - maxF1) < 1e-6;
                      const isBestFin = Math.abs(fin.beneficio_neto_pen - maxFin) < 1e-2;
                      const isSelected = selectedModelKey === item.model_key;

                      return html`
                        <tr
                          key=${item.model_key}
                          onClick=${() => setSelectedModelKey(item.model_key)}
                          className=${`cursor-pointer transition-colors ${
                            isSelected
                              ? "bg-blue-50/80 dark:bg-blue-950/30"
                              : "hover:bg-zinc-100/70 dark:hover:bg-zinc-900/60"
                          }`}
                        >
                          <td className="py-3.5 px-3.5 font-medium whitespace-nowrap">
                            <div className="flex items-center gap-1.5">
                              <span className="text-zinc-900 dark:text-zinc-100 font-semibold">
                                ${item.model_name}
                              </span>
                              ${item.is_heuristic_baseline &&
                              html`<span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                                Base
                              </span>`}
                              ${isBestFin &&
                              html`<span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 font-semibold">
                                ★ S/
                              </span>`}
                              ${isBestF1 &&
                              html`<span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-600 dark:text-blue-400 border border-blue-500/30 font-semibold">
                                ★ F1
                              </span>`}
                            </div>
                          </td>
                          <td className="py-3.5 px-2.5 text-right font-mono text-zinc-600 dark:text-zinc-300">
                            ${m.precision.toFixed(4)}
                          </td>
                          <td className="py-3.5 px-2.5 text-right">
                            <${SparkBar} value=${m.recall} colorClass="bg-emerald-500" />
                          </td>
                          <td className="py-3.5 px-2.5 text-right">
                            <${SparkBar}
                              value=${m.f1_score}
                              colorClass=${isBestF1 ? "bg-blue-500" : "bg-purple-500"}
                            />
                          </td>
                          <td className="py-3.5 px-2.5 text-right font-mono text-zinc-600 dark:text-zinc-300">
                            ${m.roc_auc.toFixed(4)}
                          </td>
                          <td className="py-3.5 px-3 text-center font-mono text-xs whitespace-nowrap">
                            <span className="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
                              TN:${cm.tn}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600 dark:text-amber-400 ml-1">
                              FP:${cm.fp}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-600 dark:text-rose-400 ml-1">
                              FN:${cm.fn}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 ml-1">
                              TP:${cm.tp}
                            </span>
                          </td>
                          <td
                            className=${`py-3.5 px-3.5 text-right font-mono font-bold whitespace-nowrap ${
                              isBestFin
                                ? "text-emerald-600 dark:text-emerald-400"
                                : "text-zinc-900 dark:text-zinc-100"
                            }`}
                          >
                            ${PEN.format(fin.beneficio_neto_pen)}
                          </td>
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>

              <!-- Tarjeta de Auditoría Detallada del Modelo Seleccionado -->
              ${selectedModelDetail &&
              html`
                <div className="xl:col-span-4 bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 space-y-4">
                  <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-3">
                    <div>
                      <span className="text-[11px] uppercase tracking-wider text-blue-500 font-semibold">
                        Auditoría de Modelo Seleccionado
                      </span>
                      <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
                        ${selectedModelDetail.model_name}
                      </h3>
                    </div>
                    <span className="font-mono text-xs px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-800">
                      Acc: ${(selectedModelDetail.metrics.accuracy * 100).toFixed(2)}%
                    </span>
                  </div>

                  <!-- Matriz de Confusión 2x2 Visual -->
                  <div className="grid grid-cols-2 gap-2.5 text-xs">
                    <div className="p-2.5 rounded-lg bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800">
                      <span className="text-zinc-500 block">Verdaderos Negativos (TN)</span>
                      <span className="font-mono text-base font-bold text-zinc-900 dark:text-zinc-100">
                        ${selectedModelDetail.confusion_matrix.tn}
                      </span>
                      <span className="text-[11px] text-zinc-400 block">Clientes aprobados OK</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/25">
                      <span className="text-amber-500 block">Falsos Positivos (FP)</span>
                      <span className="font-mono text-base font-bold text-amber-500">
                        ${selectedModelDetail.confusion_matrix.fp}
                      </span>
                      <span className="text-[11px] text-amber-400/80 block">
                        Fricción: -${PEN.format(selectedModelDetail.financial_simulation.costo_falsos_bloqueos_pen)}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/25">
                      <span className="text-rose-500 block">Falsos Negativos (FN)</span>
                      <span className="font-mono text-base font-bold text-rose-500">
                        ${selectedModelDetail.confusion_matrix.fn}
                      </span>
                      <span className="text-[11px] text-rose-400/80 block">
                        Pérdida: -${PEN.format(selectedModelDetail.financial_simulation.perdida_sufrida_pen)}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/25">
                      <span className="text-emerald-500 block">Verdaderos Positivos (TP)</span>
                      <span className="font-mono text-base font-bold text-emerald-500">
                        ${selectedModelDetail.confusion_matrix.tp}
                      </span>
                      <span className="text-[11px] text-emerald-400/80 block">
                        Ahorro: +${PEN.format(selectedModelDetail.financial_simulation.perdida_evitada_pen)}
                      </span>
                    </div>
                  </div>

                  <!-- Desglose de Validación Cruzada k=5 -->
                  <div className="p-3 rounded-lg bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 text-xs space-y-1.5">
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Validación Cruzada (k=5) F1:</span>
                      <span className="font-mono font-semibold">
                        ${selectedModelDetail.cv_5fold.f1_score_mean?.toFixed(4)} ±${selectedModelDetail.cv_5fold.f1_score_std?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Validación Cruzada (k=5) ROC-AUC:</span>
                      <span className="font-mono font-semibold">
                        ${selectedModelDetail.cv_5fold.roc_auc_mean?.toFixed(4)} ±${selectedModelDetail.cv_5fold.roc_auc_std?.toFixed(4)}
                      </span>
                    </div>
                    <div className="pt-1.5 border-t border-zinc-200 dark:border-zinc-800 flex justify-between font-bold">
                      <span>Beneficio Neto Final:</span>
                      <span className="font-mono text-emerald-500">
                        ${PEN.format(selectedModelDetail.financial_simulation.beneficio_neto_pen)}
                      </span>
                    </div>
                  </div>
                </div>
              `}
            </div>

            <!-- Gráficos Comparativos -->
            <${SupervisedCharts} models=${models} darkMode=${darkMode} />
          </section>
        `}

        <!-- =============================================================== -->
        <!-- PANEL 2: SIMULADOR DE TRANSACCIÓN EN TIEMPO REAL                -->
        <!-- =============================================================== -->
        ${(activeTab === "all" || activeTab === "panel2") &&
        html`
          <section className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 sm:p-6 shadow-sm space-y-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-5">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-500">
                  Panel 2 · Motor de Inferencia Online (POST /api/predict)
                </span>
                <h2 className="text-xl font-bold tracking-tight mt-0.5">
                  Simulador de Transacción en Tiempo Real
                </h2>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                  Modifica cualquiera de las 8 variables explicativas o elige un perfil preconfigurado para auditar el veredicto simultáneo de todos los modelos.
                </p>
              </div>

              <!-- Botones Rápidos de Escenarios -->
              <div className="flex flex-wrap gap-2">
                ${Object.entries(PRESETS).map(
                  ([key, p]) => html`
                    <button
                      key=${key}
                      type="button"
                      onClick=${() => handlePresetSelect(key)}
                      className=${`px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all ${
                        activePreset === key
                          ? key === "normal"
                            ? "bg-emerald-500 text-zinc-950 border-emerald-400 shadow-sm"
                            : key === "attack"
                            ? "bg-rose-600 text-white border-rose-500 shadow-sm"
                            : "bg-amber-500 text-zinc-950 border-amber-400 shadow-sm"
                          : "bg-zinc-100 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 border-zinc-200 dark:border-zinc-800 hover:border-zinc-600"
                      }`}
                    >
                      ${key === "normal" ? "✓ " : key === "attack" ? "⚠ " : "⚡ "}
                      ${p.label}
                    </button>
                  `
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <!-- Controles de las 8 Variables -->
              <form
                onSubmit=${(e) => {
                  e.preventDefault();
                  runPrediction(txForm, threshold);
                }}
                className="lg:col-span-5 grid grid-cols-1 sm:grid-cols-2 gap-3.5 bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4"
              >
                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    1. Monto Transacción (S/ PEN)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    min="1"
                    max="25000"
                    value=${txForm.monto_pen}
                    onChange=${(e) => handleFieldChange("monto_pen", parseFloat(e.target.value) || 1)}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    2. Hora del Día (0 a 23 h)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value=${txForm.hora_dia}
                    onChange=${(e) => handleFieldChange("hora_dia", parseInt(e.target.value, 10) || 0)}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    3. Distancia Ubicación Habitual (km)
                  </label>
                  <input
                    type="number"
                    step="0.5"
                    min="0"
                    max="5000"
                    value=${txForm.distancia_km}
                    onChange=${(e) => handleFieldChange("distancia_km", parseFloat(e.target.value) || 0)}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    4. Intentos Previos (Última 1h)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="30"
                    value=${txForm.intentos_previos_1h}
                    onChange=${(e) =>
                      handleFieldChange("intentos_previos_1h", parseInt(e.target.value, 10) || 0)}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    5. Ubicación Geográfica
                  </label>
                  <select
                    value=${txForm.es_internacional}
                    onChange=${(e) =>
                      handleFieldChange("es_internacional", parseInt(e.target.value, 10))}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                  >
                    <option value="0">0 — Nacional (Perú)</option>
                    <option value="1">1 — Internacional (Extranjero)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    6. Huella de Dispositivo
                  </label>
                  <select
                    value=${txForm.dispositivo_nuevo}
                    onChange=${(e) =>
                      handleFieldChange("dispositivo_nuevo", parseInt(e.target.value, 10))}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                  >
                    <option value="0">0 — Dispositivo Conocido</option>
                    <option value="1">1 — Dispositivo Nuevo</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    7. Ratio vs. Ticket Histórico
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.05"
                    max="25"
                    value=${txForm.ratio_monto_promedio}
                    onChange=${(e) =>
                      handleFieldChange("ratio_monto_promedio", parseFloat(e.target.value) || 0.1)}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-zinc-500 mb-1">
                    8. Antigüedad Cuenta (días)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10000"
                    value=${txForm.antiguedad_cuenta_dias}
                    onChange=${(e) =>
                      handleFieldChange("antiguedad_cuenta_dias", parseInt(e.target.value, 10) || 1)}
                    className="w-full bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="sm:col-span-2 pt-1">
                  <button
                    type="submit"
                    className="w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm transition shadow-sm"
                  >
                    Ejecutar Inferencia en Tiempo Real (POST /api/predict)
                  </button>
                </div>
              </form>

              <!-- Resultados de Inferencia por Modelo -->
              <div className="lg:col-span-7 flex flex-col justify-between space-y-4">
                ${predictionResult &&
                html`
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                    ${Object.entries(predictionResult.predictions).map(([key, item]) => {
                      const blocked = item.decision === "BLOQUEAR";
                      const pct = (item.probability * 100).toFixed(1);
                      return html`
                        <div
                          key=${key}
                          className=${`rounded-xl p-4 border transition-all ${
                            blocked
                              ? "bg-rose-500/5 dark:bg-rose-950/20 border-rose-500/35"
                              : "bg-emerald-500/5 dark:bg-emerald-950/15 border-emerald-500/30"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-bold text-zinc-900 dark:text-zinc-100 truncate">
                              ${item.model_name}
                            </span>
                            <span
                              className=${`text-[11px] font-mono font-bold px-2.5 py-0.5 rounded-full ${
                                blocked
                                  ? "bg-rose-500/20 text-rose-600 dark:text-rose-300 border border-rose-500/30"
                                  : "bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 border border-emerald-500/30"
                              }`}
                            >
                              ${item.decision}
                            </span>
                          </div>
                          <div className="mt-3">
                            <div className="flex justify-between text-xs mb-1.5">
                              <span className="text-zinc-500">Score de Riesgo:</span>
                              <span
                                className=${`font-mono font-bold ${
                                  blocked ? "text-rose-500" : "text-emerald-500"
                                }`}
                              >
                                ${pct}%
                              </span>
                            </div>
                            <div className="w-full h-2 rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                              <div
                                className=${`h-full rounded-full transition-all duration-300 ${
                                  blocked ? "bg-rose-500" : "bg-emerald-500"
                                }`}
                                style=${{ width: `${Math.max(3, Math.min(100, item.probability * 100))}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      `;
                    })}
                  </div>

                  <div className="bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs">
                    <div>
                      <span className="text-zinc-500">Resumen de Inferencia: </span>
                      <strong className="text-zinc-900 dark:text-zinc-100">
                        ${Object.values(predictionResult.predictions).filter(
                          (p) => p.decision === "BLOQUEAR"
                        ).length} de 5 modelos
                      </strong>
                      <span className="text-zinc-500"> activan bloqueo con umbral </span>
                      <span className="font-mono font-semibold text-blue-500">
                        τ = ${predictionResult.threshold.toFixed(2)}
                      </span>
                    </div>
                    <div className="font-mono text-zinc-500">
                      Monto: ${PEN.format(predictionResult.input_transaction.monto_pen)} · Hora: ${predictionResult.input_transaction.hora_dia}:00h
                    </div>
                  </div>
                `}
              </div>
            </div>
          </section>
        `}

        <!-- =============================================================== -->
        <!-- PANEL 3: AUDITORÍA NO SUPERVISADA (CLUSTERING)                  -->
        <!-- =============================================================== -->
        ${(activeTab === "all" || activeTab === "panel3") &&
        clusteringData &&
        html`
          <section className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800/90 rounded-xl p-5 sm:p-6 shadow-sm space-y-6">
            <div className="border-b border-zinc-200 dark:border-zinc-800 pb-4">
              <span className="text-xs font-semibold uppercase tracking-wider text-purple-500">
                Panel 3 · Aprendizaje No Supervisado (GET /api/clustering)
              </span>
              <h2 className="text-xl font-bold tracking-tight mt-0.5">
                Auditoría de Anomalías Transaccionales: K-Means (k=2) vs. DBSCAN (ε=1.8, min_samples=10)
              </h2>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Evaluación sin etiquetas del poder de separación geométrica (Silhouette Score) y tasa de captura de fraudes reales como outliers.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
                ${[
                  {
                    badge: "Particional por Centroides",
                    data: clusteringData.kmeans,
                    subtitle: `Cluster Normal: ${clusteringData.kmeans.cluster_sizes.normal_cluster} tx · Cluster Anómalo: ${clusteringData.kmeans.cluster_sizes.anomaly_cluster} tx`,
                  },
                  {
                    badge: "Agrupamiento por Densidad",
                    data: clusteringData.dbscan,
                    subtitle: `Clusters Densos: ${clusteringData.dbscan.n_clusters_found} · Puntos de Ruido (-1) como Anomalías`,
                  },
                ].map(
                  (item) => html`
                    <div
                      key=${item.data.algorithm}
                      className="bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 flex flex-col justify-between space-y-4"
                    >
                      <div>
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[11px] font-semibold uppercase tracking-wider text-blue-500">
                            ${item.badge}
                          </span>
                          <span className="font-mono text-xs px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-800 font-semibold">
                            Sil: ${item.data.silhouette_score.toFixed(4)}
                          </span>
                        </div>
                        <h3 className="text-base font-bold mt-2">${item.data.algorithm}</h3>
                        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                          ${item.subtitle}
                        </p>
                      </div>

                      <div className="space-y-2.5 bg-white dark:bg-[#0c0c0f] p-3.5 rounded-lg border border-zinc-200 dark:border-zinc-800 text-xs">
                        <div className="flex justify-between">
                          <span className="text-zinc-500">Outliers Totales Detectados:</span>
                          <span className="font-mono font-bold">${item.data.total_outliers_flagged} tx</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-500">Fraudes Atípicos Capturados:</span>
                          <span className="font-mono font-bold text-emerald-500">
                            ${item.data.atypical_captured} / ${clusteringData.total_atypical_frauds}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-500">Pureza del Grupo Anómalo:</span>
                          <span className="font-mono font-semibold">
                            ${(item.data.anomaly_precision * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="pt-2 border-t border-zinc-200 dark:border-zinc-800">
                          <div className="flex justify-between mb-1">
                            <span className="font-medium">Tasa de Captura de Fraude:</span>
                            <span className="font-mono font-bold text-emerald-500">
                              ${item.data.outlier_capture_pct.toFixed(2)}%
                            </span>
                          </div>
                          <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-emerald-500 rounded-full"
                              style=${{ width: `${Math.min(100, item.data.outlier_capture_pct)}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  `
                )}
              </div>

              <div className="lg:col-span-6 bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-semibold">
                      Proyección PCA 2D del Espacio Transaccional Escalado
                    </h3>
                    <p className="text-xs text-zinc-500">
                      Separación espacial entre el núcleo denso legítimo y los outliers aislados por DBSCAN
                    </p>
                  </div>
                  <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                    Varianza 2D: ${(
                      ((clusteringData.pca_explained_variance?.[0] || 0) +
                        (clusteringData.pca_explained_variance?.[1] || 0)) *
                      100
                    ).toFixed(1)}%
                  </span>
                </div>
                <${PcaScatterChart} clustering=${clusteringData} darkMode=${darkMode} />
              </div>
            </div>
          </section>
        `}
      </main>
    </div>
  `;
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(html`<${App} />`);
