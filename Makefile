# Makefile for StockPrediction project

.PHONY: help setup web run train eval test test-cov docker-build docker-up docker-up-dev docker-down clean

PYTHON=python
PORT?=8501
ADDRESS?=0.0.0.0

# Training/eval defaults (override like: make train SYMBOL=NVDA AGENT=a3c EPISODES=1000)
SYMBOL?=AAPL
AGENT?=vac
WINDOW?=30
EPISODES?=500
WORKERS?=auto
TRAIN_PERIOD?=2y
LR?=0.001
MODEL?=

help:
	@echo "Common targets:"
	@echo "  setup         Install uv and project dependencies"
	@echo "  web           Run Streamlit app on $(ADDRESS):$(PORT)"
	@echo "  run           Run CLI entry (main.py) [MODE=web|train|eval]"
	@echo "  train         Train model (override SYMBOL, AGENT, EPISODES, etc.)"
	@echo "  eval          Evaluate a saved model (set MODEL=path/to/model.pth)"
	@echo "  test          Run tests with coverage (pytest.ini configured)"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Start docker-compose"
	@echo "  docker-up-dev Start docker-compose.dev with live reload"
	@echo "  docker-down   Stop and remove compose services"
	@echo "  clean         Remove caches and coverage artifacts"

setup:
	@which uv >/dev/null 2>&1 || pip install uv
	uv sync

# Web UI via Streamlit (preferred)
web:
	uv run main.py --mode web

# Generic entrypoint to main.py; default MODE=web
MODE?=web
run:
	uv run $(PYTHON) main.py --mode $(MODE)

# Training helper with useful defaults
train:
	uv run $(PYTHON) main.py --mode train --symbol $(SYMBOL) --agent $(AGENT) --window-size $(WINDOW) --episodes $(EPISODES) --train-period $(TRAIN_PERIOD) --learning-rate $(LR) $(if $(filter-out auto,$(WORKERS)),--workers $(WORKERS),)

# Evaluation helper; require MODEL path
eval:
	@if [ -z "$(MODEL)" ]; then echo "Error: MODEL path is required. Example: make eval MODEL=data/results_vac_AAPL_20240115_143022/best_model.pth SYMBOL=AAPL AGENT=vac"; exit 1; fi
	uv run $(PYTHON) main.py --mode eval --model $(MODEL) --symbol $(SYMBOL) --agent $(AGENT)

test:
	uv run pytest

test-cov:
	uv run pytest --cov-report=term-missing

docker-build:
	docker build -t stock-prediction-app .

docker-up:
	docker compose up -d

docker-up-dev:
	docker compose -f docker-compose.dev.yml up -d

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache htmlcov coverage.xml .coverage

