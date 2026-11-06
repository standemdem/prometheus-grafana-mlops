
#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8080}"
REQUESTS="${1:-50}"

echo "Simulation de $REQUESTS requêtes sur $API_URL/predict"

for ((i=1; i<=REQUESTS; i++)); do
    HOUR=$((i % 24))
    WEEKDAY=$((i % 7))

    curl --fail --silent --show-error \
      --max-time 30 \
      -X POST "$API_URL/predict" \
      -H "Content-Type: application/json" \
      -d "{
        \"dteday\": \"2011-02-01\",
        \"season\": 1,
        \"holiday\": 0,
        \"workingday\": 1,
        \"weathersit\": 1,
        \"temp\": 0.3,
        \"atemp\": 0.3,
        \"hum\": 0.5,
        \"windspeed\": 0.2,
        \"mnth\": 2,
        \"hr\": $HOUR,
        \"weekday\": $WEEKDAY
      }" > /dev/null
done

echo "SUCCÈS : $REQUESTS prédictions effectuées."
