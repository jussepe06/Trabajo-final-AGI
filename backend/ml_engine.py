"""Motor de Machine Learning y Evaluación Comparativa (Bloque 2) para FinGuard AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from backend.data_generator import (
        DEFAULT_DATA_PATH,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        generate_transactions_dataset,
    )
except ImportError:
    from data_generator import (
        DEFAULT_DATA_PATH,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        generate_transactions_dataset,
    )

# Bloque 1: agentes heurísticos de FinGuard AI, sin sustituir el Bloque 2.
try:
    from backend.agents import ModelBasedAgent, UtilityBasedAgent
except ImportError:
    from agents import ModelBasedAgent, UtilityBasedAgent

DEFAULT_ARTIFACTS_PATH = Path(__file__).resolve().parent.parent / "data" / "finguard_artifacts.pkl"
FP_OPERATIONAL_COST_PEN = 25.0


class FraudDetectionEngine:
    """
    Motor central de entrenamiento, validación cruzada estratificada (k=5),
    evaluación comparativa con umbral dinámico, simulación financiera en soles (S/)
    y detección no supervisada de anomalías (K-Means vs. DBSCAN).
    """

    MODEL_NAMES: dict[str, str] = {
        "modo_base": "Modo Base (Regla Heurística Estática)",
        "logistic_regression": "Regresión Logística",
        "knn": "K-Nearest Neighbors (KNN)",
        "decision_tree": "Árbol de Decisión",
        "mlp": "Perceptrón Multicapa (MLP)",
    }

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        artifacts_path: Path | str = DEFAULT_ARTIFACTS_PATH,
        random_state: int = 42,
    ) -> None:
        self.data_path = Path(data_path)
        self.artifacts_path = Path(artifacts_path)
        self.random_state = random_state

        self.df: pd.DataFrame | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.X_train_scaled: np.ndarray | None = None
        self.X_test_scaled: np.ndarray | None = None

        self.scaler = StandardScaler()
        self.models: dict[str, Any] = {}
        self.test_probabilities: dict[str, np.ndarray] = {}
        self.cv_results: dict[str, dict[str, float]] = {}
        self.unsupervised_results: dict[str, Any] = {}
        self.data_quality_report: dict[str, Any] = {}
        self.is_trained: bool = False

    def _build_supervised_estimators(self) -> dict[str, Any]:
        """Instancia los clasificadores del Bloque 2 con los hiperparámetros requeridos."""
        return {
            "logistic_regression": LogisticRegression(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=1000,
            ),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "decision_tree": DecisionTreeClassifier(
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=300,
                random_state=self.random_state,
            ),
        }

    @staticmethod
    def apply_heuristic_rule(df_features: pd.DataFrame) -> np.ndarray:
        """
        Modo Base (Regla Heurística Estática):
        Bloquea (pred = 1) si monto_pen > 1500 o (hora_dia <= 4 y es_internacional == 1).
        """
        cond = (df_features["monto_pen"] > 1500.0) | (
            (df_features["hora_dia"] <= 4) & (df_features["es_internacional"] == 1)
        )
        return cond.astype(int).to_numpy()

    def load_and_prepare_data(
        self,
        force_regenerate: bool = False,
        seed: int | None = None,
        fraud_rate: float = 0.03,
    ) -> pd.DataFrame:
        """
        Carga o genera el dataset, verifica valores nulos y ejecuta la partición
        estratificada 80% train / 20% test aplicando StandardScaler sin data leakage.
        """
        if seed is not None:
            self.random_state = int(seed)

        if force_regenerate or not self.data_path.exists():
            self.df = generate_transactions_dataset(
                fraud_ratio=float(fraud_rate),
                seed=self.random_state,
                output_path=self.data_path,
            )
        else:
            self.df = pd.read_csv(self.data_path)

        missing_counts = self.df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().to_dict()
        total_missing = int(sum(missing_counts.values()))
        if total_missing > 0:
            self.df = self.df.dropna(subset=[TARGET_COLUMN]).copy()

        X = self.df[FEATURE_COLUMNS].copy()
        y = self.df[TARGET_COLUMN].astype(int).copy()

        # Partición estratificada 80% entrenamiento / 20% prueba ANTES de cualquier preprocesamiento
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=self.random_state,
        )

        # Imputación preventiva sin fuga de datos (usando medianas de X_train si existieran nulos)
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            train_medians = X_train.median()
            X_train = X_train.fillna(train_medians)
            X_test = X_test.fillna(train_medians)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # Ajuste de StandardScaler exclusivamente sobre X_train (sin data leakage)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.data_quality_report = {
            "total_rows": int(len(self.df)),
            "train_rows": int(len(self.X_train)),
            "test_rows": int(len(self.X_test)),
            "legit_count": int((y == 0).sum()),
            "fraud_count": int((y == 1).sum()),
            "fraud_ratio": round(float(y.mean()), 4),
            "missing_values_found": total_missing,
        }
        return self.df

    def _run_stratified_cv(self, n_splits: int = 5) -> dict[str, dict[str, float]]:
        """
        Ejecuta validación cruzada estratificada (k=5 folds) sobre el conjunto de
        entrenamiento usando Pipelines con StandardScaler interno por fold para
        garantizar cero fuga de datos.
        """
        assert self.X_train is not None and self.y_train is not None
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        estimators = self._build_supervised_estimators()
        cv_summary: dict[str, dict[str, float]] = {}

        # 1. Validación cruzada para el Modo Base (Regla Heurística Estática)
        base_metrics: dict[str, list[float]] = {
            "accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []
        }
        for _, val_idx in skf.split(self.X_train, self.y_train):
            X_val_fold = self.X_train.iloc[val_idx]
            y_val_fold = self.y_train.iloc[val_idx].to_numpy()
            y_pred_fold = self.apply_heuristic_rule(X_val_fold)
            base_metrics["accuracy"].append(float(accuracy_score(y_val_fold, y_pred_fold)))
            base_metrics["precision"].append(
                float(precision_score(y_val_fold, y_pred_fold, zero_division=0))
            )
            base_metrics["recall"].append(
                float(recall_score(y_val_fold, y_pred_fold, zero_division=0))
            )
            base_metrics["f1_score"].append(
                float(f1_score(y_val_fold, y_pred_fold, zero_division=0))
            )
            base_metrics["roc_auc"].append(float(roc_auc_score(y_val_fold, y_pred_fold)))

        cv_summary["modo_base"] = {
            f"{m}_mean": round(float(np.mean(vals)), 4) for m, vals in base_metrics.items()
        } | {
            f"{m}_std": round(float(np.std(vals)), 4) for m, vals in base_metrics.items()
        }

        # 2. Validación cruzada para cada modelo supervisado con Pipeline(StandardScaler, Modelo)
        for key, estimator in estimators.items():
            fold_metrics: dict[str, list[float]] = {
                "accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []
            }
            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr_fold = self.X_train.iloc[train_idx]
                y_tr_fold = self.y_train.iloc[train_idx].to_numpy()
                X_val_fold = self.X_train.iloc[val_idx]
                y_val_fold = self.y_train.iloc[val_idx].to_numpy()

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", clone(estimator)),
                ])
                pipe.fit(X_tr_fold, y_tr_fold)
                y_prob_fold = pipe.predict_proba(X_val_fold)[:, 1]
                y_pred_fold = (y_prob_fold >= 0.50).astype(int)

                fold_metrics["accuracy"].append(float(accuracy_score(y_val_fold, y_pred_fold)))
                fold_metrics["precision"].append(
                    float(precision_score(y_val_fold, y_pred_fold, zero_division=0))
                )
                fold_metrics["recall"].append(
                    float(recall_score(y_val_fold, y_pred_fold, zero_division=0))
                )
                fold_metrics["f1_score"].append(
                    float(f1_score(y_val_fold, y_pred_fold, zero_division=0))
                )
                fold_metrics["roc_auc"].append(float(roc_auc_score(y_val_fold, y_prob_fold)))

            cv_summary[key] = {
                f"{m}_mean": round(float(np.mean(vals)), 4)
                for m, vals in fold_metrics.items()
            } | {
                f"{m}_std": round(float(np.std(vals)), 4)
                for m, vals in fold_metrics.items()
            }

        return cv_summary

    def _evaluate_unsupervised(self) -> dict[str, Any]:
        """
        Compara K-Means (k=2) vs. DBSCAN (eps=1.8, min_samples=10) calculando el
        Silhouette Score y su tasa de captura de outliers sobre transacciones atípicas.
        """
        assert self.X_train_scaled is not None and self.X_test_scaled is not None
        assert self.y_train is not None and self.y_test is not None

        # Evaluamos sobre la matriz escalada completa (con StandardScaler ajustado solo en train)
        X_all_scaled = np.vstack([self.X_train_scaled, self.X_test_scaled])
        y_all = np.concatenate([self.y_train.to_numpy(), self.y_test.to_numpy()])
        total_atypical = int((y_all == 1).sum())

        # --- 1. K-Means (k=2) ---
        kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_all_scaled)
        kmeans_sil = float(silhouette_score(X_all_scaled, kmeans_labels))

        # El cluster minoritario corresponde al grupo de transacciones atípicas/anómalas
        cluster_counts = np.bincount(kmeans_labels)
        anomaly_cluster = int(np.argmin(cluster_counts))
        kmeans_is_outlier = (kmeans_labels == anomaly_cluster).astype(int)
        kmeans_captured = int(((kmeans_is_outlier == 1) & (y_all == 1)).sum())
        kmeans_capture_rate = kmeans_captured / total_atypical if total_atypical > 0 else 0.0
        kmeans_flagged = int(kmeans_is_outlier.sum())
        kmeans_precision = kmeans_captured / kmeans_flagged if kmeans_flagged > 0 else 0.0

        # --- 2. DBSCAN (eps=1.8, min_samples=10) ---
        dbscan = DBSCAN(eps=1.8, min_samples=10)
        dbscan_labels = dbscan.fit_predict(X_all_scaled)

        unique_labels = set(dbscan_labels)
        if len(unique_labels) > 1:
            dbscan_sil = float(silhouette_score(X_all_scaled, dbscan_labels))
        else:
            dbscan_sil = 0.0

        # En DBSCAN, label == -1 representa ruido / outliers (transacciones atípicas)
        dbscan_is_outlier = (dbscan_labels == -1).astype(int)
        dbscan_captured = int(((dbscan_is_outlier == 1) & (y_all == 1)).sum())
        dbscan_capture_rate = dbscan_captured / total_atypical if total_atypical > 0 else 0.0
        dbscan_flagged = int(dbscan_is_outlier.sum())
        dbscan_precision = dbscan_captured / dbscan_flagged if dbscan_flagged > 0 else 0.0
        n_clusters_dbscan = len(unique_labels - {-1})

        # Proyección PCA 2D de una muestra estratificada para visualización en el frontend
        pca = PCA(n_components=2, random_state=self.random_state)
        coords_2d = pca.fit_transform(X_all_scaled)
        rng = np.random.default_rng(self.random_state)
        fraud_indices = np.where(y_all == 1)[0]
        legit_indices = rng.choice(np.where(y_all == 0)[0], size=350, replace=False)
        sample_idx = np.concatenate([legit_indices, fraud_indices])

        projection_sample = [
            {
                "pc1": round(float(coords_2d[i, 0]), 3),
                "pc2": round(float(coords_2d[i, 1]), 3),
                "clase_real": int(y_all[i]),
                "kmeans_outlier": int(kmeans_is_outlier[i]),
                "dbscan_outlier": int(dbscan_is_outlier[i]),
            }
            for i in sample_idx
        ]

        return {
            "total_transactions_evaluated": int(len(y_all)),
            "total_atypical_frauds": total_atypical,
            "kmeans": {
                "algorithm": "K-Means (k=2)",
                "parameters": {"n_clusters": 2, "random_state": self.random_state},
                "silhouette_score": round(kmeans_sil, 4),
                "outlier_capture_rate": round(kmeans_capture_rate, 4),
                "outlier_capture_pct": round(kmeans_capture_rate * 100.0, 2),
                "atypical_captured": kmeans_captured,
                "total_outliers_flagged": kmeans_flagged,
                "anomaly_precision": round(kmeans_precision, 4),
                "cluster_sizes": {
                    "normal_cluster": int(cluster_counts[1 - anomaly_cluster]),
                    "anomaly_cluster": int(cluster_counts[anomaly_cluster]),
                },
            },
            "dbscan": {
                "algorithm": "DBSCAN (eps=1.8, min_samples=10)",
                "parameters": {"eps": 1.8, "min_samples": 10},
                "n_clusters_found": n_clusters_dbscan,
                "silhouette_score": round(dbscan_sil, 4),
                "outlier_capture_rate": round(dbscan_capture_rate, 4),
                "outlier_capture_pct": round(dbscan_capture_rate * 100.0, 2),
                "atypical_captured": dbscan_captured,
                "total_outliers_flagged": dbscan_flagged,
                "anomaly_precision": round(dbscan_precision, 4),
            },
            "pca_explained_variance": [
                round(float(v), 4) for v in pca.explained_variance_ratio_
            ],
            "projection_sample": projection_sample,
        }

    def train_all(self, force_retrain: bool = False) -> dict[str, Any]:
        """
        Entrena todos los modelos del Bloque 2, ejecuta la validación cruzada
        estratificada (k=5), evalúa el módulo no supervisado y guarda los artefactos (.pkl).
        """
        if not force_retrain and self.artifacts_path.exists():
            try:
                self._load_artifacts()
                return self.evaluate_at_threshold(threshold=0.50)
            except Exception:
                pass

        self.load_and_prepare_data(force_regenerate=False)
        assert self.X_train is not None and self.X_test is not None
        assert self.y_train is not None and self.y_test is not None
        assert self.X_train_scaled is not None and self.X_test_scaled is not None

        # 1. Validación Cruzada Estratificada (k=5 folds)
        self.cv_results = self._run_stratified_cv(n_splits=5)

        # 2. Entrenamiento de modelos supervisados sobre el 80% train escalado
        self.models = {}
        self.test_probabilities = {}

        # Modo Base sobre el conjunto de prueba (regla determinista)
        base_preds = self.apply_heuristic_rule(self.X_test)
        self.test_probabilities["modo_base"] = base_preds.astype(float)

        for key, estimator in self._build_supervised_estimators().items():
            estimator.fit(self.X_train_scaled, self.y_train.to_numpy())
            self.models[key] = estimator
            self.test_probabilities[key] = estimator.predict_proba(self.X_test_scaled)[:, 1]

        # 3. Evaluación del Módulo No Supervisado (K-Means vs. DBSCAN)
        self.unsupervised_results = self._evaluate_unsupervised()
        self.is_trained = True

        # 4. Persistencia de artefactos en data/finguard_artifacts.pkl
        self._save_artifacts()
        return self.evaluate_at_threshold(threshold=0.50)

    def _compute_metrics_and_financials(
        self,
        model_key: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        montos_pen: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        """
        Calcula métricas de clasificación, matriz de confusión y simulación financiera:
        Beneficio_Neto = Pérdida_Evitada (TP) - Pérdida_Sufrida (FN) - Costo_Falsos_Bloqueos (FP * 25)
        """
        if model_key == "modo_base":
            # La regla heurística estática no depende del umbral probabilístico
            y_pred = y_prob.astype(int)
        else:
            y_pred = (y_prob >= threshold).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, y_prob))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        perdida_evitada = float(montos_pen[(y_true == 1) & (y_pred == 1)].sum())
        perdida_sufrida = float(montos_pen[(y_true == 1) & (y_pred == 0)].sum())
        costo_falsos_bloqueos = float(fp * FP_OPERATIONAL_COST_PEN)
        beneficio_neto = float(perdida_evitada - perdida_sufrida - costo_falsos_bloqueos)
        monto_total_fraude = float(perdida_evitada + perdida_sufrida)

        return {
            "model_key": model_key,
            "model_name": self.MODEL_NAMES[model_key],
            "is_heuristic_baseline": model_key == "modo_base",
            "threshold_applied": 0.50 if model_key == "modo_base" else round(float(threshold), 2),
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
            },
            "cv_5fold": self.cv_results.get(model_key, {}),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "financial_simulation": {
                "perdida_evitada_pen": round(perdida_evitada, 2),
                "perdida_sufrida_pen": round(perdida_sufrida, 2),
                "costo_falsos_bloqueos_pen": round(costo_falsos_bloqueos, 2),
                "beneficio_neto_pen": round(beneficio_neto, 2),
                "monto_total_fraude_test_pen": round(monto_total_fraude, 2),
                "costo_unitario_fp_pen": FP_OPERATIONAL_COST_PEN,
            },
        }

    def evaluate_at_threshold(self, threshold: float = 0.50) -> dict[str, Any]:
        """
        Evalúa instantáneamente el Modo Base y todos los modelos supervisados bajo
        cualquier umbral de probabilidad (0.05 a 0.95), retornando métricas de test,
        validación cruzada k=5, matrices de confusión, impacto financiero y clustering.
        """
        if not self.is_trained:
            return self.train_all()

        threshold = float(np.clip(threshold, 0.01, 0.99))
        assert self.X_test is not None and self.y_test is not None

        y_true = self.y_test.to_numpy()
        montos_test = self.X_test["monto_pen"].to_numpy()

        supervised_comparison = []
        for key in self.MODEL_NAMES:
            y_prob = self.test_probabilities[key]
            res = self._compute_metrics_and_financials(
                model_key=key,
                y_true=y_true,
                y_prob=y_prob,
                montos_pen=montos_test,
                threshold=threshold,
            )
            supervised_comparison.append(res)

        best_financial_model = max(
            supervised_comparison,
            key=lambda item: item["financial_simulation"]["beneficio_neto_pen"],
        )
        best_f1_model = max(
            supervised_comparison,
            key=lambda item: item["metrics"]["f1_score"],
        )

        return {
            "threshold": round(threshold, 2),
            "dataset_summary": self.data_quality_report,
            "supervised_models": supervised_comparison,
            "unsupervised_clustering": self.unsupervised_results,
            "block1_agents": self.get_block1_agents_results(),
            "recommendations": {
                "best_financial_model": best_financial_model["model_name"],
                "best_financial_net_benefit_pen": best_financial_model["financial_simulation"][
                    "beneficio_neto_pen"
                ],
                "best_f1_model": best_f1_model["model_name"],
                "best_f1_score": best_f1_model["metrics"]["f1_score"],
            },
        }

    def get_block1_agents_results(self) -> dict[str, Any]:
        """Compara la regla original y dos arquitecturas de agentes en el mismo test.

        La base es sintética y no aporta account_id ni cronología: el agente basado
        en modelo evalúa señales agregadas por fila; su memoria secuencial NO se
        utiliza al no haber historiales individuales verificables.
        """
        from time import perf_counter

        if not self.is_trained:
            self.train_all()
        assert self.X_test is not None and self.y_test is not None
        X_test = self.X_test
        y_true = self.y_test.to_numpy(dtype=int)
        amounts = X_test["monto_pen"].to_numpy(dtype=float)

        evaluators = (
            ("modo_base", "Modo base: regla estática", lambda: self.apply_heuristic_rule(X_test)),
            ("agente_modelo", "Agente basado en modelo", lambda: ModelBasedAgent().predict(X_test)),
            ("agente_utilidad", "Agente basado en utilidad", lambda: UtilityBasedAgent().predict(X_test)),
        )
        results = []
        for key, name, predict in evaluators:
            start = perf_counter()
            predicted = np.asarray(predict(), dtype=int)
            prediction_ms = (perf_counter() - start) * 1000.0
            tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
            prevented = float(amounts[(y_true == 1) & (predicted == 1)].sum())
            missed = float(amounts[(y_true == 1) & (predicted == 0)].sum())
            net = prevented - missed - int(fp) * FP_OPERATIONAL_COST_PEN
            results.append({
                "agent_key": key,
                "agent_name": name,
                "metrics": {
                    "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
                    "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
                    "f1_score": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
                },
                "confusion_matrix": {
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                },
                "financial_simulation": {
                    "perdida_evitada_pen": round(prevented, 2),
                    "perdida_sufrida_pen": round(missed, 2),
                    "costo_falsos_bloqueos_pen": round(int(fp) * FP_OPERATIONAL_COST_PEN, 2),
                    "beneficio_neto_pen": round(net, 2),
                },
                "prediction_time_ms": round(prediction_ms, 3),
            })
        return {
            "description": "Bloque 1: comparación académica de estrategias heurísticas",
            "test_rows": int(len(X_test)),
            "test_fraud_count": int((y_true == 1).sum()),
            "data_origin": "Transacciones sintéticas; no son operaciones bancarias reales",
            "note": "Tiempos medidos por llamada: varían según máquina; las reglas no requieren entrenamiento. No son promedios de múltiples corridas.",
            "agents": results,
        }

    def predict_block1_agents(self, transaction: dict[str, float | int]) -> dict[str, Any]:
        """Expone decisiones heurísticas individuales, separadas de los cinco modelos originales."""
        tx_df = pd.DataFrame([{col: transaction[col] for col in FEATURE_COLUMNS}])
        baseline = int(self.apply_heuristic_rule(tx_df)[0])
        model_agent = ModelBasedAgent()
        model_state = model_agent.perceive(tx_df.iloc[0])
        model_decision = int(model_agent.predict(tx_df)[0])
        utility_agent = UtilityBasedAgent()
        risk_score = float(utility_agent.estimated_risk(tx_df)[0])
        utility_decision = int(utility_agent.predict(tx_df)[0])
        return {
            "modo_base": {"decision": "BLOQUEAR" if baseline else "APROBAR", "prediction": baseline},
            "agente_modelo": {
                "decision": "BLOQUEAR" if model_decision else "APROBAR",
                "prediction": model_decision,
                "alert_count": int(model_state.total_alerts),
            },
            "agente_utilidad": {
                "decision": "BLOQUEAR" if utility_decision else "APROBAR",
                "prediction": utility_decision,
                "heuristic_risk_score": round(risk_score, 4),
                "risk_score_is_calibrated_probability": False,
            },
        }

    def get_clustering_results(self) -> dict[str, Any]:
        """Devuelve el análisis comparativo de clustering no supervisado (K-Means vs DBSCAN)."""
        if not self.is_trained:
            self.train_all()
        return self.unsupervised_results

    def retrain_with_params(
        self,
        seed: int = 42,
        fraud_rate: float = 0.03,
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """
        Regenera el dataset con nueva semilla (seed) o tasa de fraude (fraud_rate),
        re-entrena todos los modelos supervisados y no supervisados y retorna las métricas.
        """
        self.load_and_prepare_data(
            force_regenerate=True, seed=int(seed), fraud_rate=float(fraud_rate)
        )
        assert self.X_train is not None and self.X_test is not None
        assert self.y_train is not None and self.y_test is not None
        assert self.X_train_scaled is not None and self.X_test_scaled is not None

        self.cv_results = self._run_stratified_cv(n_splits=5)
        self.models = {}
        self.test_probabilities = {
            "modo_base": self.apply_heuristic_rule(self.X_test).astype(float)
        }
        for key, estimator in self._build_supervised_estimators().items():
            estimator.fit(self.X_train_scaled, self.y_train.to_numpy())
            self.models[key] = estimator
            self.test_probabilities[key] = estimator.predict_proba(self.X_test_scaled)[:, 1]

        self.unsupervised_results = self._evaluate_unsupervised()
        self.is_trained = True
        self._save_artifacts()
        return self.evaluate_at_threshold(threshold=threshold)

    def predict_transaction(
        self, transaction: dict[str, float | int], threshold: float = 0.50
    ) -> dict[str, Any]:
        """
        Evalúa una transacción individual en vivo con el Modo Base y todos los modelos
        entrenados del Bloque 2.
        """
        if not self.is_trained:
            self.train_all()

        tx_df = pd.DataFrame([{col: transaction[col] for col in FEATURE_COLUMNS}])
        tx_scaled = self.scaler.transform(tx_df)

        predictions = {}
        base_pred = int(self.apply_heuristic_rule(tx_df)[0])
        predictions["modo_base"] = {
            "model_name": self.MODEL_NAMES["modo_base"],
            "probability": float(base_pred),
            "prediction": base_pred,
            "decision": "BLOQUEAR" if base_pred == 1 else "APROBAR",
        }

        for key, model in self.models.items():
            prob = float(model.predict_proba(tx_scaled)[0, 1])
            pred = int(prob >= threshold)
            predictions[key] = {
                "model_name": self.MODEL_NAMES[key],
                "probability": round(prob, 4),
                "prediction": pred,
                "decision": "BLOQUEAR" if pred == 1 else "APROBAR",
            }

        return {
            "input_transaction": {col: transaction[col] for col in FEATURE_COLUMNS},
            "threshold": round(float(threshold), 2),
            "predictions": predictions,
            "block1_agent_predictions": self.predict_block1_agents(transaction),
        }

    def _save_artifacts(self) -> None:
        self.artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "models": self.models,
            "test_probabilities": self.test_probabilities,
            "cv_results": self.cv_results,
            "unsupervised_results": self.unsupervised_results,
            "data_quality_report": self.data_quality_report,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "X_train_scaled": self.X_train_scaled,
            "X_test_scaled": self.X_test_scaled,
        }
        joblib.dump(payload, self.artifacts_path)

    def _load_artifacts(self) -> None:
        payload = joblib.load(self.artifacts_path)
        self.scaler = payload["scaler"]
        self.models = payload["models"]
        self.test_probabilities = payload["test_probabilities"]
        self.cv_results = payload["cv_results"]
        self.unsupervised_results = payload["unsupervised_results"]
        self.data_quality_report = payload["data_quality_report"]
        self.X_train = payload["X_train"]
        self.X_test = payload["X_test"]
        self.y_train = payload["y_train"]
        self.y_test = payload["y_test"]
        self.X_train_scaled = payload["X_train_scaled"]
        self.X_test_scaled = payload["X_test_scaled"]
        self.is_trained = True


if __name__ == "__main__":
    engine = FraudDetectionEngine()
    engine.load_and_prepare_data(force_regenerate=True)
    results = engine.train_all(force_retrain=True)

    # Verificaciones automáticas (self-check de integridad)
    summary = results["dataset_summary"]
    assert summary["total_rows"] == 5000, "El dataset debe tener exactamente 5,000 filas"
    assert summary["legit_count"] == 4850 and summary["fraud_count"] == 150, "Desbalance 97%/3% incumplido"
    assert summary["train_rows"] == 4000 and summary["test_rows"] == 1000, "Partición 80/20 incumplida"
    assert len(results["supervised_models"]) == 5, "Deben evaluarse Modo Base + 4 modelos supervisados"
    assert "kmeans" in results["unsupervised_clustering"] and "dbscan" in results["unsupervised_clustering"]

    print("\n=== RESUMEN DATASET FINGUARD AI ===")
    print(summary)

    print("\n=== VALIDACIÓN CRUZADA ESTRATIFICADA (k=5 FOLDS EN TRAIN) ===")
    cv_header = f"{'Modelo':<38} | {'CV Acc':>14} | {'CV F1-Score':>14} | {'CV ROC-AUC':>14}"
    print(cv_header)
    print("-" * len(cv_header))
    for item in results["supervised_models"]:
        cv = item["cv_5fold"]
        print(
            f"{item['model_name']:<38} | "
            f"{cv['accuracy_mean']:.4f} ±{cv['accuracy_std']:.4f} | "
            f"{cv['f1_score_mean']:.4f} ±{cv['f1_score_std']:.4f} | "
            f"{cv['roc_auc_mean']:.4f} ±{cv['roc_auc_std']:.4f}"
        )

    print("\n=== COMPARATIVA SUPERVISADA (TEST SET | UMBRAL = 0.50) ===")
    header = (
        f"{'Modelo':<38} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | "
        f"{'F1':>6} | {'AUC':>6} | {'TN':>4} {'FP':>3} {'FN':>3} {'TP':>3} | {'Beneficio Neto (S/)':>19}"
    )
    print(header)
    print("-" * len(header))
    for item in results["supervised_models"]:
        m = item["metrics"]
        cm = item["confusion_matrix"]
        fin = item["financial_simulation"]
        print(
            f"{item['model_name']:<38} | "
            f"{m['accuracy']:>6.4f} | {m['precision']:>6.4f} | {m['recall']:>6.4f} | "
            f"{m['f1_score']:>6.4f} | {m['roc_auc']:>6.4f} | "
            f"{cm['tn']:>4d} {cm['fp']:>3d} {cm['fn']:>3d} {cm['tp']:>3d} | "
            f"S/ {fin['beneficio_neto_pen']:>15,.2f}"
        )

    print("\n=== MÓDULO NO SUPERVISADO (CLUSTERING DE ANOMALÍAS) ===")
    for key in ["kmeans", "dbscan"]:
        u = results["unsupervised_clustering"][key]
        print(
            f"- {u['algorithm']:<32}: Silhouette = {u['silhouette_score']:.4f} | "
            f"Captura Outliers = {u['outlier_capture_pct']:.2f}% ({u['atypical_captured']}/{results['unsupervised_clustering']['total_atypical_frauds']}) | "
            f"Total Anomalías Detectadas = {u['total_outliers_flagged']}"
        )

