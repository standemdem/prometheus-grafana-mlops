.PHONY: all stop evaluation fire-alert traffic

all:
	docker compose up --build -d

stop:
	docker compose down

evaluation:
	docker compose run --rm evaluation

fire-alert:
	docker compose exec -T bike-api python -c "import sys; sys.path.insert(0, '/app'); exec(open('/app/fire_alert.py').read())"

traffic:
	bash scripts/generate_traffic.sh 50