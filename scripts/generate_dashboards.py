import json
from pathlib import Path

OUTPUT_DIR = Path("deployment/grafana/dashboards")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASOURCE = {"type": "prometheus", "uid": "prometheus"}


def panel(title, expression, panel_id, x, y, width=12, height=8,
          unit="short", panel_type="timeseries", thresholds=None):
    result = {
        "id": panel_id,
        "title": title,
        "type": panel_type,
        "datasource": DATASOURCE,
        "gridPos": {
            "x": x,
            "y": y,
            "w": width,
            "h": height
        },
        "targets": [
            {
                "refId": "A",
                "expr": expression,
                "legendFormat": "{{instance}}"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": thresholds or [
                        {"color": "green", "value": None}
                    ]
                }
            },
            "overrides": []
        },
        "options": {}
    }

    if panel_type == "stat":
        result["options"] = {
            "reduceOptions": {
                "values": False,
                "calcs": ["lastNotNull"],
                "fields": ""
            },
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto"
        }

    return result


def dashboard(title, uid, panels):
    return {
        "id": None,
        "uid": uid,
        "title": title,
        "tags": ["bike-sharing", "mlops", "monitoring"],
        "timezone": "browser",
        "schemaVersion": 41,
        "version": 1,
        "refresh": "15s",
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "panels": panels
    }


api_panels = [
    panel(
        "API - Requêtes par seconde",
        'sum(rate(api_requests_total{endpoint!="/metrics"}[5m]))',
        1, 0, 0, unit="reqps"
    ),
    panel(
        "API - Latence P95",
        'histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket{endpoint!="/metrics"}[5m])))',
        2, 12, 0, unit="s"
    ),
    panel(
        "API - Taux d'erreur HTTP",
        '100 * sum(rate(api_requests_total{status_code=~"5.."}[5m])) / clamp_min(sum(rate(api_requests_total[5m])), 0.000001)',
        3, 0, 8, unit="percent"
    ),
    panel(
        "API - Requêtes par endpoint",
        'sum by (endpoint) (rate(api_requests_total{endpoint!="/metrics"}[5m]))',
        4, 12, 8, unit="reqps"
    )
]

model_panels = [
    panel("Modèle - RMSE", "model_rmse_score", 1, 0, 0,
          panel_type="stat"),
    panel("Modèle - MAE", "model_mae_score", 2, 12, 0,
          panel_type="stat"),
    panel("Modèle - R²", "model_r2_score", 3, 0, 8,
          panel_type="stat"),
    panel("Modèle - MAPE (%)", "model_mape_score", 4, 12, 8,
          unit="percent", panel_type="stat"),
    panel(
        "Evidently - Dérive des données",
        "evidently_data_drift_detected_status",
        5, 0, 16, width=24,
        panel_type="stat",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "red", "value": 1}
        ]
    ),
    panel(
        "Évolution du RMSE",
        "model_rmse_score",
        6, 0, 24, width=24
    )
]

infra_panels = [
    panel(
        "Infrastructure - CPU utilisé (%)",
        '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])))',
        1, 0, 0, unit="percent"
    ),
    panel(
        "Infrastructure - RAM utilisée (%)",
        '100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)',
        2, 12, 0, unit="percent"
    ),
    panel(
        "Infrastructure - Espace disque utilisé (%)",
        '100 * (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})',
        3, 0, 8, unit="percent"
    ),
    panel(
        "Infrastructure - Mémoire disponible",
        "node_memory_MemAvailable_bytes",
        4, 12, 8, unit="bytes"
    )
]

dashboards = {
    "API-dashboard.json": dashboard(
        "Bike Sharing - API Monitoring",
        "bike-api-monitoring",
        api_panels
    ),
    "Model-dashboard.json": dashboard(
        "Bike Sharing - Model Monitoring",
        "bike-model-monitoring",
        model_panels
    ),
    "Infra-dashboard.json": dashboard(
        "Bike Sharing - Infrastructure Monitoring",
        "bike-infra-monitoring",
        infra_panels
    )
}

for filename, content in dashboards.items():
    destination = OUTPUT_DIR / filename
    destination.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Dashboard généré : {destination}")
