# Monitoring MLOps — Bike Sharing

> **Examen DataScientest · Prometheus & Grafana**  
> Prédiction du nombre de vélos partagés, suivi des performances du modèle et détection de dérive des données.

## Vue d'ensemble

Ce projet met en place une chaîne de monitoring pour un modèle de régression entraîné sur le jeu de données **UCI Bike Sharing**. Une API FastAPI expose les prédictions et les évaluations ; **Evidently** calcule les indicateurs de performance et de dérive ; **Prometheus** collecte les métriques ; **Grafana** les visualise et déclenche des alertes.

### Architecture

```text
                    ┌────────────────────────────┐
                    │ UCI Bike Sharing (hour.csv) │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ FastAPI · bike-api :8080       │
                   │ RandomForestRegressor         │
                   │ /predict · /evaluate · /metrics│
                   └───────────────┬───────────────┘
                                   │
                       Evidently + prometheus_client
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Prometheus :9090                │◄── Node Exporter :9100
                  │ Scraping + alerte disponibilité │
                  └────────────────┬───────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Grafana :3000                   │
                  │ 3 dashboards + alerte RMSE      │
                  └────────────────────────────────┘
```

## Démarrage rapide

### Prérequis

- Docker et Docker Compose (`docker compose`)
- GNU Make
- Bash et `curl` pour la génération de trafic
- Accès à Internet lors du téléchargement du dataset UCI

Aucun environnement Python local n'est nécessaire.

```bash
# Construire et démarrer les services
make

# Initialiser les métriques du modèle
make evaluation

# Générer du trafic HTTP
make traffic
```

| Service | URL |
|---|---|
| API FastAPI | http://localhost:8080 |
| Documentation interactive | http://localhost:8080/docs |
| Métriques Prometheus de l'API | http://localhost:8080/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Node Exporter | http://localhost:9100/metrics |

**Grafana :** identifiants configurés pour une installation neuve : `admin` / `admin` (à modifier pour tout usage hors examen).

> **Important :** les métriques ML sont initialisées à zéro au démarrage de l'API. Exécuter `make evaluation` pour calculer les premières valeurs. Les données persistantes du modèle, de Prometheus et de Grafana sont conservées dans des volumes Docker lors de `make stop`.

## Modèle de prédiction

- **Algorithme :** `RandomForestRegressor` (100 arbres, `random_state=42`).
- **Entraînement :** observations de **janvier 2011**.
- **Cible :** `cnt`, nombre total de vélos loués.
- **Variables numériques :** `temp`, `atemp`, `hum`, `windspeed`, `mnth`, `hr`, `weekday`.
- **Variables catégorielles :** `season`, `holiday`, `workingday`, `weathersit`.
- **Persistance :** modèle sérialisé avec `joblib` dans un volume Docker ; réutilisé aux démarrages suivants s'il existe déjà.

## API FastAPI

### `POST /predict` — Prédiction

Reçoit une observation et renvoie une prédiction de `cnt`.

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "dteday": "2011-02-01",
    "temp": 0.3,
    "atemp": 0.3,
    "hum": 0.5,
    "windspeed": 0.2,
    "mnth": 2,
    "hr": 12,
    "weekday": 3,
    "season": 1,
    "holiday": 0,
    "workingday": 1,
    "weathersit": 1
  }'
```

Le champ `dteday` est obligatoire et doit respecter le format `YYYY-MM-DD`.

### `POST /evaluate` — Évaluation

Reçoit un nom de période et une liste d'observations contenant les variables explicatives et la cible réelle `cnt`. Le modèle produit les prédictions, puis **Evidently** calcule les métriques de régression et recherche une dérive par rapport aux données de référence de janvier 2011.

### `GET /metrics` — Export Prometheus

Expose les métriques au format compatible Prometheus.

## Métriques collectées

| Métrique | Type | Utilité |
|---|---|---|
| `api_requests_total` | Counter | Nombre de requêtes par endpoint, méthode et statut HTTP |
| `api_request_duration_seconds` | Histogram | Temps de réponse, notamment la latence P95 |
| `model_rmse_score` | Gauge | Erreur quadratique moyenne, racine carrée (RMSE) |
| `model_mae_score` | Gauge | Erreur absolue moyenne (MAE) |
| `model_r2_score` | Gauge | Coefficient de détermination R² |
| `model_mape_score` | Gauge | Erreur absolue moyenne en pourcentage (MAPE) |
| `evidently_data_drift_detected_status` | Gauge | Dérive détectée (`1`) ou non (`0`) |

**Pourquoi ces métriques supplémentaires ?** Le **MAPE** apporte une lecture relative de l'erreur de prédiction ; l'indicateur **Data Drift** signale une évolution statistique des variables d'entrée, même si le RMSE reste sous le seuil d'alerte.

Les métriques de modèle sont mises à jour à chaque évaluation. Elles sont réinitialisées au redémarrage du processus API jusqu'à la prochaine évaluation.

## Dashboards Grafana

Trois dashboards sont **provisionnés automatiquement à partir de fichiers JSON**, dans le dossier Grafana `Bike Sharing Monitoring`.

| Dashboard | Principaux indicateurs |
|---|---|
| **API Performance** | Taux de requêtes, latence P95, taux d'erreur HTTP |
| **Model Performance & Drift** | RMSE, MAE, R², MAPE, dérive Evidently |
| **Infrastructure Overview** | CPU, mémoire RAM, espace disque via Node Exporter |

Fichiers : `deployment/grafana/dashboards/`.  
Provisioning : `deployment/grafana/provisioning/dashboards/dashboards.yaml`.

## Alertes

### Prometheus — Disponibilité de l'API

Règle **`BikeAPIUnavailable`** :

```promql
up{job="bike-api"} == 0
```

La condition doit persister **1 minute**. Configuration : `deployment/prometheus/rules/alert_rules.yml`.

### Grafana — Dégradation du modèle

Règle **`BikeModelHighRMSE`**, créée et testée dans l'interface Grafana, puis exportée en YAML pour être reproduite sur une nouvelle installation.

- **Métrique surveillée :** `model_rmse_score`
- **Condition :** RMSE > **25**
- **Provisioning :** `deployment/grafana/provisioning/alerting/alert_rules.yaml`

L'état **Firing** est visible après le prochain cycle d'évaluation de Grafana lorsque la condition est remplie.

## Scénarios de test

### 1. Évaluation normale

```bash
make evaluation
```

Le conteneur d'évaluation exécute `src/evaluation/run_evaluation.py` et envoie les observations à `/evaluate`.

**Résultats observés lors des tests :**

| Indicateur | Valeur |
|---|---:|
| Période | `week1_february` |
| Observations | 235 |
| RMSE | **17,5234** |
| MAPE | **29,3653 %** |
| Data Drift | **Oui** |

### 2. Déclenchement volontaire de l'alerte

```bash
make fire-alert
```

Le script `src/api/fire_alert.py` soumet **100 observations modifiées**, avec une cible `cnt` augmentée artificiellement de 500, afin de provoquer une dégradation du RMSE.

**Résultat observé :** RMSE ≈ **505,4916**, supérieur au seuil de **25**. L'alerte testée est **`BikeModelHighRMSE`**.

Pour rétablir les métriques de l'évaluation normale :

```bash
make evaluation
```

L'alerte revient à l'état normal après le prochain cycle d'évaluation Grafana.

### 3. Génération de trafic HTTP

```bash
make traffic
```

Exécute `scripts/generate_traffic.sh` et envoie **50 requêtes POST** à `/predict`, en faisant varier l'heure et le jour de la semaine.

**Résultat observé :** 50 réponses **HTTP 200**.

Vérification :

```bash
curl -fsS http://localhost:8080/metrics \
  | grep 'api_requests_total{endpoint="/predict"'
```

Les éventuels anciens compteurs HTTP 422 correspondent à des requêtes de test invalides ; ils sont conservés par Prometheus jusqu'au redémarrage de l'API.

## Commandes disponibles

| Commande | Action |
|---|---|
| `make` ou `make all` | Construire et démarrer API, Prometheus, Grafana et Node Exporter |
| `make stop` | Arrêter les services sans supprimer les volumes |
| `make evaluation` | Évaluer le modèle et actualiser les métriques |
| `make fire-alert` | Simuler une erreur élevée pour déclencher l'alerte RMSE |
| `make traffic` | Envoyer 50 requêtes à `/predict` |

## Organisation du projet

```text
.
├── Makefile
├── README.md
├── EXAM_INSTRUCTIONS.md
├── docker-compose.yml
├── deployment/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/alert_rules.yml
│   └── grafana/
│       ├── dashboards/
│       │   ├── API-dashboard.json
│       │   ├── Model-dashboard.json
│       │   └── Infra-dashboard.json
│       └── provisioning/
│           ├── alerting/alert_rules.yaml
│           ├── dashboards/dashboards.yaml
│           └── datasources/datasources.yaml
├── src/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── fire_alert.py
│   │   └── requirements.txt
│   └── evaluation/
│       ├── Dockerfile
│       ├── run_evaluation.py
│       └── requirements.txt
├── scripts/
│   ├── generate_dashboards.py
│   └── generate_traffic.sh
└── tests/
    └── test_evidently.py
```

## Reproduction sur une autre machine

```bash
make                 # Démarrer l'environnement
make evaluation      # Initialiser les métriques ML
make traffic         # Alimenter les métriques API
make fire-alert      # Tester l'alerte RMSE
make evaluation      # Restaurer les métriques normales
make stop            # Arrêter les services
```

Le premier démarrage nécessite le téléchargement du dataset UCI. Les dashboards et les règles d'alerte sont inclus dans le projet afin de faciliter la reproductibilité.

---

*Projet d'examen Liora — Prometheus & Grafana / MLOps.*
