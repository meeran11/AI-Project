# Intelligent Vehicle Routing & Dynamic Pricing System
### FAST-NUCES Karachi · AI/ML Project 2026

**Students:** Muhammad Masab (23K-0833) & Meeran Uz Zaman (23K-0039)  
**Course:** Artificial Intelligence / Machine Learning  
**Instructor:** Abdullah Shaikh

---

## Project Overview

A dual-task AI/ML system implementing:

| Task | Algorithm | Purpose |
|------|-----------|---------|
| **Task 1** | A* Search | Optimal vehicle routing across Karachi city graph |
| **Task 2** | Gradient Boosting Regression | Dynamic daily rental price prediction |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (6,500 records)
python generate_dataset.py

# 3. Run both tasks (A* + ML training)
python main.py

# 4. Open interactive dashboard
start smartroute-ui.html   # Modern UI Dashboard
# OR
start dashboard.html       # Original dashboard
```

---

## 🎨 User Interface

### SmartRoute AI Dashboard (`smartroute-ui.html`)
A modern, professional UI featuring:

- **🚗 Route Planning**: Interactive A* pathfinding visualization
- **💰 Price Prediction**: Live ML-based pricing calculator  
- **📊 Analytics**: Feature importance and sensitivity analysis
- **🎯 Real-time Updates**: Dynamic calculations and animations
- **📱 Responsive Design**: Works on desktop and mobile devices

### Features
- **Interactive City Graph**: Click to explore Karachi's 18 locations
- **A* Algorithm Demo**: Watch the search algorithm in action
- **Price Predictor**: Input parameters to get instant price predictions
- **Model Analytics**: View feature importance and performance metrics
- **Modern UI/UX**: Dark theme with smooth animations and transitions

---

## Task 1 — A* Informed Search

### Algorithm
**A\*** evaluates each node using:
```
f(n) = g(n) + h(n)
```
- `g(n)` — actual cost from start to current node (km)
- `h(n)` — Euclidean (straight-line) distance heuristic to goal

The heuristic is **admissible** (never overestimates) and **consistent**, guaranteeing optimality.

### City Graph
- **18 nodes** — major Karachi localities (Clifton, Defence, Gulshan, DHA, Karachi Airport, etc.)
- **31 bidirectional edges** — road connections with real approximate distances (km)
- Coordinates sourced from actual geographic data

### Available Nodes
```
Clifton, Defence, Korangi, Gulshan, North Nazimabad, Saddar,
Lyari, Orangi, SITE, Karachi Airport, Malir, Landhi,
Baldia, Kemari, Gulberg, Federal B Area, Shah Faisal, DHA
```

### Sample Run
```
Start: Clifton → Goal: Gulshan
Optimal Path: Clifton → Saddar → Gulshan (15.50 km)
Nodes Explored: 3 / 18
Search Time: 0.243 ms
```

### Outputs
- `astar_result.png` — high-quality graph visualization
- `astar_result.json` — machine-readable path + exploration data

---

## Task 2 — Dynamic Pricing ML

### Dataset
- **6,500 synthetic records** generated with real-world patterns
- **12 features**: vehicle category, day of week, weather, season, holiday status, competitor pricing, utilization rate, etc.
- Price range: $26–$220/day, mean $84.98

### Models Trained

| Model | MAE | RMSE | CV RMSE | R² |
|-------|-----|------|---------|-----|
| Random Forest | $4.10 | $5.38 | $5.71 | 0.9755 |
| **Gradient Boosting ★** | **$3.16** | **$4.04** | **$4.23** | **0.9862** |

**Winner: Gradient Boosting** (200 estimators, lr=0.08, max_depth=5)

### Top Feature Importances
1. `competitor_base_price` — 38.2% (market anchoring)
2. `vehicle_category_Luxury` — 13.4%
3. `utilization_rate` — 9.1%
4. `season_Summer` — 5.8%
5. `is_weekend` — 4.7%

### Sample Predictions
| Vehicle | Day | Weather | Predicted |
|---------|-----|---------|-----------|
| Economy | Friday | Sunny | $57.38 |
| Luxury | Monday | Rainy | $146.53 |
| SUV | Saturday | Snowy | $138.96 |
| Compact | Wednesday | Cloudy | $52.08 |

### Outputs
- `pricing_model.pkl` — serialized trained pipeline
- `pricing_results.png` — evaluation plots (actual vs predicted, feature importance, model comparison)
- `pricing_results.json` — metrics + feature importances + sample predictions

---

## Interactive Dashboard (`dashboard.html`)

Open in any modern browser — no server required.

### Features
- **A* Tab**: Interactive canvas graph with all 18 Karachi nodes. Select any start/goal, run A*, animate step-by-step exploration with 220ms frame delay, view exploration log with g(n) costs.
- **Dynamic Pricing Tab**: Live price predictor (adjust all 8 input features, instant prediction), feature importance bars, model comparison table, multi-line sensitivity chart (price vs day of week for all vehicle types).
- **Project Overview Tab**: Problem statement, architecture diagram, tech stack, sample predictions.

---

## File Structure
```
vehicle-routing-system/
├── main.py                  # Main pipeline (both tasks)
├── generate_dataset.py      # Synthetic dataset generator
├── requirements.txt         # Python dependencies
├── dashboard.html           # Interactive web dashboard
├── README.md               # This file
│
├── rental_pricing_dataset.csv   # Generated dataset (6,500 rows)
├── pricing_model.pkl            # Trained Gradient Boosting pipeline
│
├── astar_result.png         # A* visualization
├── astar_result.json        # A* path + exploration data
├── pricing_results.png      # ML evaluation plots
└── pricing_results.json     # ML metrics + predictions
```

---

## Technology Stack

| Library | Version | Use |
|---------|---------|-----|
| Python | 3.11+ | Core language |
| scikit-learn | ≥1.3 | ML pipeline, preprocessing, cross-validation |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical operations |
| matplotlib | ≥3.7 | Visualization |
| networkx | ≥3.1 | Graph operations (A* visualization) |

---

## Evaluation Metrics

### A* Correctness
- Optimal path guaranteed (admissible heuristic)
- Explored nodes ≪ total nodes (efficiency)

### ML Metrics
- **MAE** (Mean Absolute Error) — average price error in dollars
- **RMSE** (Root Mean Squared Error) — penalizes large errors
- **R²** (Coefficient of Determination) — variance explained (0.9862 = 98.62%)
- **5-fold Cross-Validation RMSE** — generalisation check

---

*Submitted: April 2026 | FAST-NUCES Karachi*
