# 📢 Sentiment Classifier as a Service

A tiny but complete end-to-end AI/ML project that classifies text reviews as Positive or Negative.  
Built with Python, scikit-learn, FastAPI, and Docker, and deployable on Google Cloud (Vertex AI + Cloud Run).

---

## 🚀 Project overview

- **Problem:** Businesses need to quickly analyze customer feedback (reviews, tweets, support tickets) to understand sentiment.
- **Solution:** A sentiment analysis service with a REST API that returns a class label and well-calibrated probabilities.
- **Tech stack:**
  - **ML:** scikit-learn pipelines (TF–IDF + Logistic Regression), optional spaCy preprocessing, XGBoost variant for comparison
  - **API:** FastAPI with Pydantic schemas and OpenAPI docs
  - **Infra:** Docker for portability; Cloud Run for serverless serving; Vertex AI for training jobs
- **Status:** Core model trained, API implemented, tests passing, Dockerized, ready for Cloud Run.

---

## 📂 Project structure

. ├── app │ ├── init.py │ ├── main.py # FastAPI app (predict endpoint, schemas) │ ├── model.py # Model load/inference helpers │ ├── schemas.py # Pydantic request/response models │ └── spacy_preprocessor.py # SpacyPreprocessor (fit/transform) ├── models/ # Serialized artifacts (vectorizer, model) ├── reports/ # Metrics, plots, and analysis ├── tests/ # API, model, preprocessing, docker tests ├── notebooks/ # Training/EDA notebooks ├── benchmark.py # CLI/Script to benchmark models (optional) ├── Dockerfile ├── requirements.txt └── README.md

---

## ⚡ Quickstart (local)

- **Python environment:**
  1. **Create venv:** `python -m venv .venv && source .venv/bin/activate`
  2. **Install deps:** `pip install -r requirements.txt`
  3. **Download spaCy model (first time):** `python -m spacy download en_core_web_sm`

- **Run API (dev):**
  - `uvicorn app.main:app --reload`
  - Open interactive docs: http://127.0.0.1:8000/docs

- **Sample request (curl):**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "I absolutely loved this movie!"}'
    ```
- **Example response:**
	```Json
	{
	  "prediction": "positive",
	  "probabilities": {
	    "negative": 0.03,
	    "positive": 0.97
	  }
	}
	```

## 🧪 Testing and quality
- **Run tests:** `pytest tests/`
- **Coverage:** `pytest --cov=app --cov-report=term-missing`
- **What's covered:** API contract, edge cases (empty/emoji/non-ASCII/long), probability
	sums, invalid payloads, model loading, preprocessing behavior, basic performance.

Tip: if spaCy/Click deprecation warnings are noisy, add a pytest.ini with
```config
	[pytest]
	filterwarnings =
	    ignore::DeprecationWarning:typer.*
	    ignore::DeprecationWarning:spacy.*
	```

## 🐳 Docker usage
- **Build image:** `docker build -t sentiment-service .`
- **Run container:** `docker run -p 8000:8000 sentiment-service`
- **Healthcheck:**
	```Bash
	curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"text":"Great product!"}'
	```

Note: The server in the container starts via the image's CMD. Adjust the Dockerfile if
you need GPU/CPU variations or custom workers.

## ☁️  Deploy on Google Cloud Run (serverless)
- **Prerequisites:**
	- gcloud auth: `gcloud auth login && gcloud config set project YOUR_PROJECT_ID`
	- **Artifact Registry repo:** `gcloud artifacts repositories create sentiment-repo 
		--repository-format=docker --location=YOUR_REGION`

- **Build and push:**
	```Bash
	gcloud builds submit --tag YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/sentiment-repo/sentiment-service:latest
	```

- **Deploy to Cloud Run:**
	```Bash
	gcloud run deploy sentiment-service \
	  --image YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/sentiment-repo/sentiment-service:latest \
	  --platform managed \
	  --region YOUR_REGION \
	  --allow-unauthenticated \
	  --memory 512Mi \
	  --concurrency 8 \
	  --cpu 1
	```
- **Test public endpoint:**
	```Bash
	curl -X POST https://YOUR_CLOUD_RUN_URL/predict \
	  -H "Content-Type: application/json" \
	  -d '{"text":"Shipping was fast and support was helpful!"}'
	```

## 🧠 Training and benchmarking
- **Artifacts:** `models/` holds serialized pipelines (e.g., `tfidf_baseline.pkl`, 
	`logreg_baseline.pkl`, `vectorizer_v2.pkl`, `sentiment_model_v2.pkl`, 
	`sentiment_model_xgb_v2.pkl`) plus metrics JSON.

- **Compare models:** See `reports/README.md` for metrics, plots (reports/roc.png,
	reports/reliability.png), and interpretation.

- **Re-run benchmanrks:** Use `notebooks/` or `benchmark.py` (ensure dataset availability
	and update paths as needed).

## 🗺️  Roadmap
- **Short term:** Add request batching, logging/trace IDs, input validation error
messages, and rate limiting headers.
- **Next:** Cloud Build + Cloud Run deploy pipeline, scheduled retraining, BigQuery backing
store for logs.
- **Stretch:** Multi-class sentiment, emotion tags, language detection, and promptable
explanations.

## 🏷️  License and acknowledgments
- **License:** MIT (or your choice).
- **Acknowledgments:** IMDB reviews dataset and spaCy English model (`en_core_web_sm`).
