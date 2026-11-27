"""API REST en FastAPI y servidor de archivos estáticos para FinGuard AI."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.ml_engine import FraudDetectionEngine

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

engine = FraudDetectionEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Inicializa y carga en memoria el motor FraudDetectionEngine una sola vez al arrancar."""
    engine.train_all(force_retrain=False)
    yield


app = FastAPI(
    title="FinGuard AI — Fraud & Anomaly Detection API",
    description=(
        "Sistema desacoplado de detección de fraude financiero y anomalías transaccionales "
        "(Bloque 1 — agentes heurísticos y Bloque 2 — Machine Learning)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionInput(BaseModel):
    monto_pen: float = Field(..., gt=0, description="Monto de la transacción en Soles (PEN)")
    hora_dia: int = Field(..., ge=0, le=23, description="Hora del día (0 a 23)")
    distancia_km: float = Field(..., ge=0, description="Distancia desde ubicación habitual (km)")
    intentos_previos_1h: int = Field(..., ge=0, description="Transacciones en la última hora")
    es_internacional: int = Field(..., ge=0, le=1, description="0 = Perú, 1 = Extranjero")
    dispositivo_nuevo: int = Field(..., ge=0, le=1, description="0 = Conocido, 1 = Nuevo")
    ratio_monto_promedio: float = Field(..., gt=0, description="Ratio frente al ticket histórico")
    antiguedad_cuenta_dias: int = Field(..., ge=1, description="Días desde apertura de cuenta")
    threshold: float = Field(0.50, ge=0.05, le=0.95, description="Umbral de probabilidad (0.05 a 0.95)")


class RetrainRequest(BaseModel):
    seed: int = Field(42, ge=1, le=999999, description="Semilla aleatoria determinista")
    fraud_rate: float = Field(0.03, ge=0.01, le=0.25, description="Proporción de fraude (ej. 0.03 = 3%)")
    threshold: float = Field(0.50, ge=0.05, le=0.95, description="Umbral de evaluación tras reentrenar")


@app.get("/api/metrics")
def get_metrics(
    threshold: float = Query(0.50, ge=0.05, le=0.95, description="Umbral de decisión en vivo")
) -> dict[str, Any]:
    """
    Devuelve la tabla comparativa completa en tiempo real (Modo Base, Regresión Logística,
    KNN, Árbol de Decisión, MLP) recalculando Precision, Recall, F1-Score, Matriz de Confusión
    y Beneficio Neto (S/) dinámicamente según el parámetro threshold.
    """
    return engine.evaluate_at_threshold(threshold=threshold)


@app.get("/api/agents", tags=["Bloque 1 · Agentes"])
def get_block1_agents() -> dict[str, Any]:
    """Compara la regla base, el agente basado en modelo y el agente basado en utilidad.

    Las métricas se calculan con las mismas 1,000 transacciones de prueba.
    Los datos y los costos son simulados; no se usan transacciones bancarias reales.
    """
    return engine.get_block1_agents_results()


@app.post("/api/agents/predict", tags=["Bloque 1 · Agentes"])
def predict_with_block1_agents(payload: TransactionInput) -> dict[str, Any]:
    """Devuelve las tres decisiones heurísticas para una transacción simulada.

    El campo `threshold` pertenece a los modelos del Bloque 2: no afecta
    las decisiones heurísticas de este endpoint.
    """
    tx_dict = payload.model_dump(exclude={"threshold"})
    return engine.predict_block1_agents(tx_dict)


@app.get("/api/clustering")
def get_clustering() -> dict[str, Any]:
    """
    Devuelve los resultados comparativos de K-Means (k=2) vs DBSCAN (eps=1.8, min_samples=10):
    Silhouette Score, cantidad de outliers detectados y tasa de captura de fraude.
    """
    return engine.get_clustering_results()


@app.post("/api/predict")
def predict_single_transaction(payload: TransactionInput) -> dict[str, Any]:
    """
    Recibe las 8 variables de una transacción simulada y el threshold actual;
    retorna la probabilidad de fraude y decisión (BLOQUEAR / APROBAR) de cada modelo.
    """
    tx_dict = payload.model_dump()
    threshold = tx_dict.pop("threshold", 0.50)
    return engine.predict_transaction(tx_dict, threshold=threshold)


@app.post("/api/retrain")
def retrain_models(
    payload: RetrainRequest | None = Body(default=None),
    seed: int | None = Query(default=None, ge=1, le=999999),
    fraud_rate: float | None = Query(default=None, ge=0.01, le=0.25),
    threshold: float | None = Query(default=None, ge=0.05, le=0.95),
) -> dict[str, Any]:
    """
    Permite re-ejecutar el entrenamiento con una nueva semilla (seed) o porcentaje
    de fraude (fraud_rate) para demostraciones en vivo (vía JSON body o query params).
    """
    eff_seed = seed if seed is not None else (payload.seed if payload else 42)
    eff_fraud_rate = (
        fraud_rate if fraud_rate is not None else (payload.fraud_rate if payload else 0.03)
    )
    eff_threshold = (
        threshold if threshold is not None else (payload.threshold if payload else 0.50)
    )
    return engine.retrain_with_params(
        seed=eff_seed, fraud_rate=eff_fraud_rate, threshold=eff_threshold
    )


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
