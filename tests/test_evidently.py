import json
import pandas as pd

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import RMSE, MAE, R2Score, MAPE
from evidently.presets import DataDriftPreset

reference_df = pd.DataFrame({
    "temp": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "cnt": [10, 20, 30, 40, 50, 60],
    "prediction": [12, 19, 32, 38, 53, 59],
})

current_df = pd.DataFrame({
    "temp": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "cnt": [15, 25, 35, 45, 55, 65],
    "prediction": [14, 27, 33, 46, 52, 68],
})

definition = DataDefinition(
    numerical_columns=["temp", "cnt", "prediction"],
    regression=[
        Regression(
            target="cnt",
            prediction="prediction"
        )
    ]
)

reference = Dataset.from_pandas(
    reference_df,
    data_definition=definition
)

current = Dataset.from_pandas(
    current_df,
    data_definition=definition
)

report = Report(metrics=[
    RMSE(),
    MAE(),
    R2Score(),
    MAPE(),
    DataDriftPreset(columns=["temp"]),
])

snapshot = report.run(
    current_data=current,
    reference_data=reference
)

result = snapshot.dict()

print("=== CLES DU SNAPSHOT ===")
print(list(result.keys()))

print("\n=== RESULTATS DES METRIQUES ===")
for metric in result.get("metrics", []):
    print(json.dumps(metric, indent=2, default=str)[:2500])
