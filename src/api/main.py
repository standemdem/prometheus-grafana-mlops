import datetime
import time
import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import RandomForestRegressor

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score, MAPE
from evidently.presets import DataDriftPreset

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèle et données de référence chargés au démarrage.
model = None
reference_data = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, reference_data

    logger.info("Initialisation du modèle Bike Sharing")

    full_data = _process_data(_fetch_data())

    january_data = full_data.loc[
        REFERENCE_START:REFERENCE_END
    ].copy()

    if january_data.empty:
        raise RuntimeError("Aucune donnée de référence pour janvier 2011")

    model, reference_data = _train_and_predict_reference_model(
        january_data
    )

    logger.info(
        "Modèle prêt avec %s observations de référence",
        len(reference_data)
    )

    yield

    logger.info("Arrêt de l'API Bike Sharing")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Prometheus Metrics Definitions ---

# Registre dédié aux métriques de l'application.
registry = CollectorRegistry()

API_REQUESTS = Counter(
    "api_requests_total",
    "Total number of HTTP requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

API_DURATION = Histogram(
    "api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

MODEL_RMSE = Gauge(
    "model_rmse_score",
    "Root Mean Squared Error of the regression model",
    registry=registry,
)

MODEL_MAE = Gauge(
    "model_mae_score",
    "Mean Absolute Error of the regression model",
    registry=registry,
)

MODEL_R2 = Gauge(
    "model_r2_score",
    "R2 score of the regression model",
    registry=registry,
)

# MAPE : permet de suivre l'erreur relative du modèle
# Elle complète RMSE et MAE, exprimés en nombre de vélos
MODEL_MAPE = Gauge(
    "model_mape_score",
    "Mean Absolute Percentage Error of the regression model",
    registry=registry,
)

# Indique si Evidently détecte une dérive des données
# 0 = aucune dérive détectée ; 1 = dérive détectée
# Cette métrique permettra de déclencher une alerte Grafana
MODEL_DRIFT = Gauge(
    "evidently_data_drift_detected_status",
    "Data drift status: 1 if detected, 0 otherwise",
    registry=registry,
)

# --- Global Variables for Model and Data ---
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']

ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

MODEL_PATH = Path("/app/models/bike_random_forest.joblib")

REFERENCE_START = "2011-01-01 00:00:00"
REFERENCE_END = "2011-01-31 23:00:00"

# --- Data Ingestion and Preparation Functions ---
def _fetch_data() -> pd.DataFrame:
    """Télécharge le dataset Bike Sharing UCI (hour.csv)."""
    logger.info("Téléchargement du dataset Bike Sharing UCI")

    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("hour.csv") as csv_file:
            data = pd.read_csv(
                csv_file,
                parse_dates=["dteday"]
            )

    logger.info("Dataset téléchargé : %s observations", len(data))
    return data


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données et construit un index temporel horaire."""
    data = raw_data.copy()

    data["hr"] = data["hr"].astype(int)

    data.index = pd.DatetimeIndex(
        data["dteday"]
        + pd.to_timedelta(data["hr"], unit="h")
    )

    data = data.sort_index()

    logger.info("Dataset préparé : %s observations", len(data))
    return data


def _train_and_predict_reference_model(
    reference_data: pd.DataFrame,
) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """
    Entraîne le modèle sur janvier 2011 et génère
    les prédictions de référence.
    """
    X_reference = reference_data[ALL_MODEL_FEATS]
    y_reference = reference_data[TARGET]

    if MODEL_PATH.exists():
        logger.info("Chargement du modèle existant : %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
    else:
        logger.info("Entraînement du RandomForestRegressor")

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_reference, y_reference)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        logger.info("Modèle sauvegardé : %s", MODEL_PATH)

    reference_with_predictions = reference_data.copy()

    reference_with_predictions[PREDICTION] = model.predict(
        X_reference
    )

    return model, reference_with_predictions

def _extract_evidently_metrics(snapshot) -> dict:
    """
    Extrait les résultats du Snapshot Evidently 0.7.23.

    RMSE et R2Score retournent des valeurs numériques.
    MAE et MAPE retournent un dictionnaire avec 'mean'.
    """
    results = {}

    for metric in snapshot.dict().get("metrics", []):
        metric_type = metric.get("config", {}).get("type", "")
        value = metric.get("value")

        if metric_type == "evidently:metric_v2:RMSE":
            results["rmse"] = float(value)

        elif metric_type == "evidently:metric_v2:MAE":
            results["mae"] = float(value["mean"])

        elif metric_type == "evidently:metric_v2:R2Score":
            results["r2score"] = float(value)

        elif metric_type == "evidently:metric_v2:MAPE":
            results["mape"] = float(value["mean"])

        elif metric_type == "evidently:metric_v2:DriftedColumnsCount":
            results["drift_detected"] = int(
                value["share"] >= 0.5
            )

    return results

# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int


# --- API Endpoints ---
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Mesure le nombre et la durée des requêtes HTTP."""

    start_time = time.perf_counter()

    endpoint = request.url.path
    method = request.method
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        duration = time.perf_counter() - start_time

        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code),
        }

        API_REQUESTS.labels(**labels).inc()
        API_DURATION.labels(**labels).observe(duration)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: BikeSharingInput):
    """Prédit le nombre de vélos partagés."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible"
        )

    features = pd.DataFrame(
        [input_data.model_dump()]
    )[ALL_MODEL_FEATS]

    prediction = model.predict(features)[0]

    return PredictionOutput(
        predicted_count=float(prediction)
    )

@app.get("/metrics")
async def metrics():
    """Expose les métriques au format Prometheus."""

    return Response(
        content=generate_latest(registry),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(payload: EvaluationData):
    """Évalue les performances et la dérive du modèle."""

    if model is None or reference_data is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle ou les données de référence sont indisponibles"
        )

    if not payload.data:
        raise HTTPException(
            status_code=400,
            detail="Aucune donnée fournie pour l'évaluation"
        )

    try:
        current_df = pd.DataFrame(payload.data)

        required_columns = ALL_MODEL_FEATS + [TARGET]

        missing = [
            col for col in required_columns
            if col not in current_df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Colonnes manquantes : {missing}"
            )

        current_df = current_df.copy()

        current_df[PREDICTION] = model.predict(
            current_df[ALL_MODEL_FEATS]
        )

        reference_df = reference_data.copy()

        # Déclaration explicite des colonnes pour Evidently.
        definition = DataDefinition(
            numerical_columns=NUM_FEATS + [TARGET, PREDICTION],
            categorical_columns=CAT_FEATS,
            regression=[
                Regression(
                    target=TARGET,
                    prediction=PREDICTION
                )
            ]
        )

        reference_dataset = Dataset.from_pandas(
            reference_df[ALL_MODEL_FEATS + [TARGET, PREDICTION]],
            data_definition=definition
        )

        current_dataset = Dataset.from_pandas(
            current_df[ALL_MODEL_FEATS + [TARGET, PREDICTION]],
            data_definition=definition
        )

        report = Report(
            metrics=[
                RMSE(),
                MAE(),
                R2Score(),
                MAPE(),
                DataDriftPreset(columns=ALL_MODEL_FEATS)
            ]
        )

        snapshot = report.run(
            current_data=current_dataset,
            reference_data=reference_dataset
        )

        results = _extract_evidently_metrics(snapshot)

        required_results = [
            "rmse", "mae", "r2score",
            "mape", "drift_detected"
        ]

        missing_results = [
            key for key in required_results
            if key not in results
        ]

        if missing_results:
            raise RuntimeError(
                f"Métriques Evidently manquantes : {missing_results}"
            )

        # Mise à jour des Gauges Prometheus.
        MODEL_RMSE.set(results["rmse"])
        MODEL_MAE.set(results["mae"])
        MODEL_R2.set(results["r2score"])
        MODEL_MAPE.set(results["mape"])
        MODEL_DRIFT.set(results["drift_detected"])

        logger.info(
            "Évaluation %s : RMSE=%.4f, MAE=%.4f, R2=%.4f, MAPE=%.4f, drift=%s",
            payload.evaluation_period_name,
            results["rmse"],
            results["mae"],
            results["r2score"],
            results["mape"],
            results["drift_detected"]
        )

        return EvaluationReportOutput(
            message=f"Evaluation completed for {payload.evaluation_period_name}",
            rmse=results["rmse"],
            mae=results["mae"],
            r2score=results["r2score"],
            mape=results["mape"],
            drift_detected=results["drift_detected"],
            evaluated_items=len(current_df)
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Erreur lors de l'évaluation")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}"
        ) from exc