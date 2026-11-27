
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd

FEATURES = (
    "monto_pen", "hora_dia", "distancia_km", "intentos_previos_1h",
    "es_internacional", "dispositivo_nuevo", "ratio_monto_promedio",
    "antiguedad_cuenta_dias",
)
FP_COST_PEN = 25.0


def _check_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Valida las ocho columnas que utiliza el proyecto original."""
    missing = [name for name in FEATURES if name not in transactions.columns]
    if missing:
        raise ValueError(f"Faltan columnas del dataset: {missing}")
    if transactions.loc[:, FEATURES].isna().any().any():
        raise ValueError("No se permiten nulos para evaluar agentes")
    return transactions


def baseline_predict(transactions: pd.DataFrame) -> np.ndarray:
    """Misma regla base del ml_engine.py: sirve como referencia verificable."""
    df = _check_features(transactions)
    blocked = (df["monto_pen"] > 1500) | (
        (df["hora_dia"] <= 4) & (df["es_internacional"] == 1)
    )
    return blocked.astype(int).to_numpy()


@dataclass(frozen=True)
class ObservedRiskState:
    """Estado interno reconstruido a partir de señales actuales e históricas."""
    activity_alert: bool
    context_alert: bool
    account_alert: bool
    total_alerts: int


class ModelBasedAgent:
    """Agente basado en modelo: percibe señales y construye estado de riesgo.

    El estado puede persistir por cuenta mediante `decide(account_id, tx)`.
    En la evaluación por lotes el dataset no tiene ID de cliente y por ello
    `predict()` utiliza un estado reconstruido de los indicadores históricos.
    """

    def __init__(self, memory_size: int = 3) -> None:
        if memory_size < 1:
            raise ValueError("memory_size debe ser >= 1")
        self.memory_size = memory_size
        self.state_by_account: dict[str, deque[ObservedRiskState]] = {}

    @staticmethod
    def perceive(transaction: pd.Series | dict) -> ObservedRiskState:
        tx = transaction
        activity = int(tx["intentos_previos_1h"]) >= 3
        context = (
            int(tx["dispositivo_nuevo"]) == 1
            and (float(tx["distancia_km"]) >= 60 or int(tx["es_internacional"]) == 1)
        )
        account = (
            float(tx["ratio_monto_promedio"]) >= 2.5
            or int(tx["antiguedad_cuenta_dias"]) <= 60
        )
        return ObservedRiskState(
            activity_alert=activity,
            context_alert=context,
            account_alert=account,
            total_alerts=int(activity) + int(context) + int(account),
        )

    def decide(self, account_id: str, transaction: pd.Series | dict) -> int:
        """Actualiza memoria por cuenta y decide a partir de las 3 últimas señales.

        Solo usar con ID real/sintético y secuencia temporal ordenada. No usar
        filas aleatorias del dataset como si pertenecieran al mismo cliente.
        """
        state = self.perceive(transaction)
        history = self.state_by_account.setdefault(
            str(account_id), deque(maxlen=self.memory_size)
        )
        history.append(state)
        alerts = (
            any(previous.activity_alert for previous in history),
            any(previous.context_alert for previous in history),
            any(previous.account_alert for previous in history),
        )
        return int(sum(alerts) >= 2)

    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        """Evalúa snapshots sin mezclar los estados de cuentas desconocidas."""
        df = _check_features(transactions)
        states = (self.perceive(row) for _, row in df.iterrows())
        return np.fromiter((int(s.total_alerts >= 2) for s in states), dtype=int, count=len(df))


class UtilityBasedAgent:
    """Agente basado en utilidad: escoge la decisión de menor pérdida esperada.

    Su estimación de riesgo es un puntaje heurístico NO calibrado; no se debe
    comunicar como probabilidad bancaria validada. No consulta etiquetas reales.
    """

    def __init__(self, false_positive_cost_pen: float = FP_COST_PEN) -> None:
        if false_positive_cost_pen < 0:
            raise ValueError("El costo de falso positivo no puede ser negativo")
        self.false_positive_cost_pen = float(false_positive_cost_pen)

    @staticmethod
    def estimated_risk(transactions: pd.DataFrame) -> np.ndarray:
        df = _check_features(transactions)
        # Señales observables, pesos fijos definidos como hipótesis del equipo.
        score = (
            -6.0
            + 1.8 * (df["intentos_previos_1h"] >= 3).astype(float)
            + 1.5 * (df["dispositivo_nuevo"] == 1).astype(float)
            + 1.5 * (df["ratio_monto_promedio"] >= 2.5).astype(float)
            + 0.9 * (df["es_internacional"] == 1).astype(float)
            + 0.8 * (df["distancia_km"] >= 80).astype(float)
            + 0.7 * (df["antiguedad_cuenta_dias"] <= 90).astype(float)
            + 0.5 * (df["hora_dia"] <= 4).astype(float)
        )
        return 1.0 / (1.0 + np.exp(-score.to_numpy(dtype=float)))

    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        df = _check_features(transactions)
        risk = self.estimated_risk(df)
        amount = df["monto_pen"].to_numpy(dtype=float)
        # Aprobar: potencial pérdida p * monto. Bloquear: costo esperado
        # de reclamación (1-p) * S/25. Se elige la pérdida menor.
        cost_approve = risk * amount
        cost_block = (1.0 - risk) * self.false_positive_cost_pen
        return (cost_block < cost_approve).astype(int)


def compare_agents(transactions: pd.DataFrame, actual_class: pd.Series | np.ndarray) -> pd.DataFrame:
    """Compara sobre el MISMO conjunto: matriz, F1 y beneficio económico simulado.

    Para evaluación final pasar solo el conjunto TEST, nunca el total de
    transacciones mezclado con el conjunto de entrenamiento.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    df = _check_features(transactions)
    actual = np.asarray(actual_class, dtype=int)
    if len(df) != len(actual):
        raise ValueError("Las transacciones y sus etiquetas no tienen igual longitud")
    amount = df["monto_pen"].to_numpy(dtype=float)
    methods = {
        "Regla estática": baseline_predict(df),
        "Agente basado en modelo": ModelBasedAgent().predict(df),
        "Agente basado en utilidad": UtilityBasedAgent().predict(df),
    }
    records = []
    for name, predicted in methods.items():
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        prevented = float(amount[(actual == 1) & (predicted == 1)].sum())
        missed = float(amount[(actual == 1) & (predicted == 0)].sum())
        net = prevented - missed - fp * FP_COST_PEN
        records.append({
            "Agente": name,
            "Accuracy": round(float(accuracy_score(actual, predicted)), 4),
            "Recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
            "F1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Beneficio_neto_S": round(net, 2),
        })
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    dataset_path = Path(__file__).resolve().parent.parent / "data" / "transacciones.csv"
    dataset = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.loc[:, FEATURES], dataset["clase"], test_size=0.20,
        stratify=dataset["clase"], random_state=42,
    )
    print("Comparación exploratoria del Bloque 1 (1000 transacciones de prueba):")
    print(compare_agents(X_test, y_test).to_string(index=False))
