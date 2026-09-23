# Food Delivery Performance & Operational Analytics

> **IBM Data Analytics Internship Project**  
> Pure Data Analytics — No Machine Learning | No Generative AI | No Prediction Models

---

## Project Overview

This project delivers a comprehensive **operational data analytics** solution for a food delivery service.
It analyses ~50,000 delivery orders to identify performance bottlenecks, operational patterns, and
business improvement opportunities across delivery time, traffic, rider performance, restaurant efficiency,
and environmental conditions.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Food Delivery Time Prediction Dataset](https://www.kaggle.com/datasets/dharmendrapandit12/food-delivery-time-prediction-dataset) |
| Records | ~50,000 |
| Columns | 24 |
| File | `data/raw/Food_Delivery_Time_Prediction.csv` |

---

## Project Structure

```
food_delivery_analytics/
│
├── data/
│   ├── raw/
│   │   └── Food_Delivery_Time_Prediction.csv      ← original dataset
│   └── processed/
│       ├── cleaned_data.csv                       ← output of cleaning pipeline
│       └── quality_report.txt                     ← data quality log
│
├── notebooks/
│   └── EDA.ipynb                                  ← standalone EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                             ← loads & validates raw CSV
│   ├── cleaner.py                                 ← 8-step cleaning pipeline
│   ├── feature_engineer.py                        ← 9 derived columns
│   ├── kpi_calculator.py                          ← 10 KPI computations
│   ├── statistical_analysis.py                    ← grouped stats, correlations, BQ functions
│   └── charts.py                                  ← reusable Plotly chart factory
│
├── pages/
│   ├── 01_Executive_Overview.py
│   ├── 02_Delivery_Performance.py
│   ├── 03_Traffic_Route_Analysis.py
│   ├── 04_Rider_Restaurant_Performance.py
│   └── 05_Operational_Environmental.py
│
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py
│   ├── test_feature_engineer.py
│   └── test_kpi_calculator.py
│
├── app.py                                         ← Streamlit entry point
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md                              ← Academic project report
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Navigate to the project directory
cd food_delivery_analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
# Copy Food_Delivery_Time_Prediction.csv into: data/raw/
```

---

## Running the Dashboard

```bash
# From inside the food_delivery_analytics/ directory:
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

**Dashboard Pages:**
| Page | Description |
|------|-------------|
| Home (app.py) | Project overview + top KPIs |
| 01 Executive Overview | All 10 KPIs + trend charts |
| 02 Delivery Performance | BQ2, BQ3, BQ4, BQ5 analysis |
| 03 Traffic & Route | BQ4, BQ5, BQ10 zone analysis |
| 04 Rider & Restaurant | BQ6, BQ7, BQ8 performance |
| 05 Operational & Environmental | BQ9 + weather, cuisine, festival |

---

## Running the EDA Notebook

```bash
# From inside the food_delivery_analytics/ directory:
jupyter notebook notebooks/EDA.ipynb
```

Then select **Kernel → Restart & Run All** to execute all cells.

---

## Running Tests

```bash
# From inside the food_delivery_analytics/ directory:
python -m pytest tests/ -v
```

Expected output: All tests pass with no errors.

---

## 10 KPIs

| KPI | Description |
|-----|-------------|
| Total Orders | Total number of delivery orders |
| Avg Delivery Time | Mean delivery time in minutes |
| Median Delivery Time | Median delivery time in minutes |
| Avg Preparation Time | Mean restaurant preparation time |
| Avg Road Distance | Mean road distance in km |
| Avg Rider Rating | Mean rider rating (1–5 scale) |
| Avg Restaurant Rating | Mean restaurant rating (1–5 scale) |
| Avg Speed | Mean rider speed in km/h |
| Weekend Orders % | Percentage of orders placed on weekends |
| Festival Orders % | Percentage of orders placed on festival days |

---

## 10 Business Questions Answered

| # | Question | Page |
|---|----------|------|
| BQ1 | What is the average delivery time across all orders? | Executive Overview |
| BQ2 | Which day of the week has the highest average delivery time? | Delivery Performance |
| BQ3 | Which time of day experiences the highest delivery delays? | Delivery Performance |
| BQ4 | How does traffic level affect delivery time? | Delivery Performance / Traffic |
| BQ5 | How does road distance affect delivery time? | Delivery Performance / Traffic |
| BQ6 | Which vehicle type has the lowest average delivery time? | Rider & Restaurant |
| BQ7 | How does restaurant preparation time affect total delivery time? | Rider & Restaurant |
| BQ8 | How do rider experience and rating relate to delivery performance? | Rider & Restaurant |
| BQ9 | How do weather conditions affect delivery time? | Operational & Environmental |
| BQ10 | Which pickup and drop-off zones have the highest delivery time? | Traffic & Route |

---

## Technology Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| Pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical computation |
| Plotly | 5.18+ | Interactive visualisation |
| Streamlit | 1.32+ | Dashboard framework |
| Matplotlib/Seaborn | Latest | EDA notebook charts |
| Pytest | 7.4+ | Unit testing |
| Jupyter | Latest | EDA notebook |

> ⚠️ **No scikit-learn, TensorFlow, LangChain, OpenAI, or any ML/AI library is used.**

---

## Key Findings (Summary)

> *All findings are computed programmatically from the actual dataset.*

- Delivery times vary significantly by traffic level, distance category, and weather conditions.
- High-traffic conditions and long-distance routes are the primary drivers of delivery delay.
- Rider experience and rating show measurable correlation with delivery efficiency.
- Festival and weekend orders tend to have different delivery time profiles vs regular orders.
- Certain pickup → drop-off zone corridors are consistently slower than others.

*(Detailed findings are available in the PROJECT_REPORT.md and the dashboard Key Insights panels.)*

---

## Academic Context

This project was developed as part of an **IBM Data Analytics Internship**.

- **Approach:** Pure descriptive and exploratory data analytics
- **No ML/AI:** No predictive models, no generative AI, no recommendation systems
- **Reproducibility:** All analysis steps are documented and coded; the pipeline runs end-to-end from raw CSV
- **Academic standard:** Code is well-commented, tested, and structured for professional review
