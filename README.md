# E-commerce Shipping Delay Prediction

Predict whether an e-commerce order will be delayed during shipping, using customer and order analytics. This project performs end-to-end EDA, data preprocessing, feature encoding, and benchmarks several machine learning models to identify the best performer.

## Dataset

The analysis uses the **Customer Analytics** dataset (`Train.csv`) from Kaggle ([prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)).

**Target variable:** `Reached.on.Time_Y.N` is inverted so that:
- `1` = shipping delay
- `0` = on-time delivery

## Contents of the Notebook

1. **Libraries & data loading**
2. **Exploratory Data Analysis (EDA)**
   - Dataset shape and statistical summary
   - Missing value & duplicate checks (none found)
   - Categorical feature analysis
   - Correlation heatmap
   - Target class distribution (countplot)
   - Boxplots for numeric features (outlier analysis)
3. **Preprocessing**
   - Train/validation/test split (stratified, to prevent data leakage)
   - `OrdinalEncoder` for `Product_importance` (low → medium → high order)
   - `OneHotEncoder` for `Warehouse_block`, `Mode_of_Shipment`, `Gender`
   - `StandardScaler` for numeric columns
   - Everything wrapped in a `ColumnTransformer`
4. **Modeling**
   - Random Forest (depth tuning)
   - XGBoost (threshold tuning + GridSearchCV + feature importance)
   - LightGBM
5. **Evaluation**
   - Confusion matrices
   - Classification reports
   - ROC-AUC comparison

## Key Observations

- **Categorical features:** `Mode_of_Shipment`, `Warehouse_block`, and `Product_importance` each have 3 categories; `Product_importance` has a natural ordering so it is ordinal-encoded, while the others are one-hot encoded.
- **Outliers:** `Prior_purchases` and `Discount_offered` contain outliers. These are kept because tree-based models are naturally robust to extreme values, whereas gradient-based algorithms (e.g., Logistic Regression, SVM, KNN) would be sensitive to them.
- **Class imbalance handling:** `scale_pos_weight` is tuned in XGBoost to rebalance class weighting.
- **Threshold tuning:** XGBoost decision threshold is optimized on validation data to keep recall ≥ 0.80 while maximizing precision.

## Best Metrics per Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.6872 | 0.7012 | 0.8113 | 0.7522 | 0.7485 |
| XGBoost | 0.6918 | 0.7041 | 0.8195 | 0.7574 | 0.7531 |
| LightGBM | 0.6855 | 0.6998 | 0.8097 | 0.7507 | 0.7462 |

## Summary

**XGBoost** achieves the best overall performance across all metrics, with the highest accuracy (0.6918), F1-Score (0.7574), and ROC-AUC (0.7531). It also delivers the best recall (0.8195), meaning it correctly captures the most delayed shipments — particularly valuable for prioritizing at-risk orders. Random Forest is a close runner-up, while LightGBM trails slightly.

The pipeline prioritizes the preprocessing step (fit on train, transform on test) to avoid data leakage, and uses threshold tuning with recall-based selection to maximize detection of delays.

## Application

The trained XGBoost model is served as a production app made of two components:

| Component | Description | Default URL |
|-----------|-------------|-------------|
| **API** | FastAPI service that loads the model and exposes prediction endpoints | `http://localhost:8000` |
| **Frontend** | Streamlit web UI that collects shipment details and calls the API | `http://localhost:8501` |

### FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service status and model name |
| `GET` | `/health` | Health check — returns status, model version, and whether the model loaded |
| `POST` | `/predict` | Predicts whether a shipment will be delayed from input features |

**Example `/health` response:**

```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_loaded": true
}
```

**Example `/predict` request body** (all fields required):

```json
{
  "Warehouse_block": "B",
  "Mode_of_Shipment": "Ship",
  "Customer_care_calls": 4,
  "Customer_rating": 4,
  "Cost_of_the_Product": 150.0,
  "Prior_purchases": 3,
  "Product_importance": "low",
  "Discount_offered": 12.5,
  "Weight_in_gms": 850.0,
  "Gender": "M"
}
```

**Example `/predict` response:**

```json
{
  "prediction": "On Time",
  "is_delayed": false,
  "delay_probability": 0.1649,
  "threshold_applied": 0.37
}
```

### Running Locally (without Docker)

```bash
# 1. Backend (FastAPI)
uvicorn api.app:app --host 0.0.0.0 --port 8000

# 2. Frontend (Streamlit) — in a second terminal
streamlit run frontend/app_frontend.py
```

## Containerization (Docker)

Both components are packaged as Docker images and published to Docker Hub under the `prasheshadk` account:

- `prasheshadk/ecommerce-shipping-delay-api:latest`
- `prasheshadk/ecommerce-shipping-delay-frontend:latest`

### Run with Docker Compose

```bash
docker compose up -d
docker compose down   # stop
```

### Run from Docker Hub (images only, no source needed)

```yaml
# docker-compose.yml
services:
  api:
    image: prasheshadk/ecommerce-shipping-delay-api:latest
    ports:
      - "8000:8000"

  frontend:
    image: prasheshadk/ecommerce-shipping-delay-frontend:latest
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000/predict
    depends_on:
      - api
```

```bash
docker compose up -d
```

### Run with plain Docker

```bash
docker network create shipping-net

docker run -d --name api --network shipping-net -p 8000:8000 \
  prasheshadk/ecommerce-shipping-delay-api:latest

docker run -d --name frontend --network shipping-net -p 8501:8501 \
  -e API_URL=http://api:8000/predict \
  prasheshadk/ecommerce-shipping-delay-frontend:latest
```

Then open `http://localhost:8501` in your browser.

> **Note:** In Docker, `API_URL` must use the container name (`http://api:8000/predict`) so the frontend and API can communicate over the shared Docker network. When running locally without Docker, it falls back to `http://localhost:8000/predict`.

## Requirements

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- lightgbm
- fastapi, uvicorn, joblib, pydantic (API)
- streamlit, requests (frontend)
- Docker (optional, for running via containers)
