import io
import sys
import zipfile

import pandas as pd
import requests


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike+sharing+dataset.zip"
)

API_URL = "http://localhost:8080/evaluate"

FEATURES = [
    "temp", "atemp", "hum", "windspeed",
    "mnth", "hr", "weekday",
    "season", "holiday", "workingday", "weathersit"
]


def main():
    print("Téléchargement du dataset Bike Sharing...")

    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("hour.csv") as file:
            df = pd.read_csv(file)

    # Échantillon de février 2011.
    samples = df[
        (df["dteday"] >= "2011-02-01")
        & (df["dteday"] <= "2011-02-07")
    ].head(100).copy()

    if samples.empty:
        raise RuntimeError("Aucune observation disponible")

    # Augmentation volontaire de la cible pour simuler
    # une dégradation importante des performances.
    samples["cnt"] = samples["cnt"] + 500

    payload = {
        "evaluation_period_name": "fire_alert_test",
        "data": samples[
            FEATURES + ["cnt", "dteday"]
        ].to_dict(orient="records")
    }

    print(f"Envoi de {len(samples)} observations modifiées...")

    result = requests.post(
        API_URL,
        json=payload,
        timeout=180
    )

    result.raise_for_status()

    evaluation = result.json()

    rmse = evaluation["rmse"]

    print(f"RMSE obtenu : {rmse:.4f}")
    print(f"Seuil d'alerte : 25")

    if rmse <= 25:
        print("ÉCHEC : le seuil n'a pas été dépassé")
        sys.exit(1)

    print("SUCCÈS : le seuil RMSE a été dépassé")
    print("Attendre la prochaine évaluation Grafana pour vérifier l'état Firing.")


if __name__ == "__main__":
    main()