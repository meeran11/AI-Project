"""
main.py  ─  Intelligent Vehicle Routing & Dynamic Pricing System
================================================================
FAST-NUCES AI/ML Project | Muhammad Masab (23K-0833) & Meeran Uz Zaman (23K-0039)

Runs TWO tasks end-to-end:
  Task 1 → A* search on a synthetic Karachi city graph
  Task 2 → Random Forest regression for dynamic rental pricing

Usage:
  python main.py                    # runs both tasks
  python main.py --task astar       # A* only
  python main.py --task pricing     # ML only
  python main.py --task astar --src "Clifton" --dst "Gulshan"
"""

import argparse
import heapq
import math
import json
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ════════════════════════════════════════════════════════════════════════════
# TASK 1 ─ A* Search Algorithm on Karachi City Graph
# ════════════════════════════════════════════════════════════════════════════

# Approximate lat/lon coordinates of major Karachi localities
KARACHI_NODES = {
    "Clifton":          (24.8138, 67.0300),
    "Defence":          (24.7960, 67.0565),
    "Korangi":          (24.8312, 67.1253),
    "Gulshan":          (24.9216, 67.0908),
    "North Nazimabad":  (24.9466, 67.0383),
    "Saddar":           (24.8607, 67.0099),
    "Lyari":            (24.8670, 67.0002),
    "Orangi":           (24.9553, 66.9997),
    "SITE":             (24.9163, 66.9989),
    "Karachi Airport":  (24.9065, 67.1609),
    "Malir":            (24.8958, 67.1989),
    "Landhi":           (24.8516, 67.1900),
    "Baldia":           (24.8922, 66.9633),
    "Kemari":           (24.8256, 66.9816),
    "Gulberg":          (24.9001, 67.0700),
    "Federal B Area":   (24.9299, 67.0650),
    "Shah Faisal":      (24.8819, 67.1384),
    "DHA":              (24.7833, 67.0714),
}

# Road connections [node_a, node_b, distance_km]
KARACHI_EDGES = [
    ("Clifton",         "Defence",          4.2),
    ("Clifton",         "Saddar",           5.8),
    ("Clifton",         "Kemari",           9.1),
    ("Defence",         "DHA",              3.5),
    ("Defence",         "Korangi",          9.3),
    ("DHA",             "Korangi",          11.0),
    ("Korangi",         "Landhi",           4.7),
    ("Korangi",         "Shah Faisal",      6.2),
    ("Korangi",         "Malir",            8.4),
    ("Landhi",          "Malir",            5.5),
    ("Landhi",          "Shah Faisal",      4.0),
    ("Shah Faisal",     "Karachi Airport",  7.1),
    ("Malir",           "Karachi Airport",  6.8),
    ("Saddar",          "Lyari",            2.3),
    ("Saddar",          "Gulshan",          9.7),
    ("Saddar",          "Gulberg",          7.2),
    ("Saddar",          "Federal B Area",   9.0),
    ("Lyari",           "Kemari",           4.6),
    ("Lyari",           "Baldia",           7.1),
    ("Kemari",          "Baldia",           9.3),
    ("Baldia",          "Orangi",           5.2),
    ("Orangi",          "SITE",             4.8),
    ("SITE",            "North Nazimabad",  6.5),
    ("SITE",            "Gulberg",          5.3),
    ("North Nazimabad", "Federal B Area",   3.9),
    ("North Nazimabad", "Gulshan",          6.1),
    ("Gulshan",         "Federal B Area",   4.2),
    ("Gulshan",         "Karachi Airport",  8.5),
    ("Gulshan",         "Gulberg",          6.0),
    ("Federal B Area",  "Karachi Airport",  9.2),
    ("Gulberg",         "Shah Faisal",      8.8),
]


def euclidean_heuristic(a, b, nodes):
    """Straight-line distance heuristic (admissible)."""
    lat1, lon1 = nodes[a]
    lat2, lon2 = nodes[b]
    # Approximate km per degree at Karachi latitude
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians(lat1))
    return math.sqrt(dlat**2 + dlon**2)


def build_graph(nodes, edges):
    """Build adjacency list."""
    G = {n: [] for n in nodes}
    for a, b, w in edges:
        G[a].append((b, w))
        G[b].append((a, w))
    return G


def astar(graph, nodes, start, goal):
    """
    A* search.
    Returns: (path, total_cost, explored_order, g_scores_at_explore)
    """
    open_heap = []          # (f, g, node)
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from    = {}
    g_score      = {n: float("inf") for n in nodes}
    g_score[start] = 0.0
    f_score      = {n: float("inf") for n in nodes}
    f_score[start] = euclidean_heuristic(start, goal, nodes)

    open_set     = {start}
    explored     = []       # order nodes were finalized

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current not in open_set:
            continue
        open_set.discard(current)
        explored.append((current, g))

        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, g_score[goal], explored

        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f_score[neighbor]   = tentative_g + euclidean_heuristic(neighbor, goal, nodes)
                heapq.heappush(open_heap, (f_score[neighbor], tentative_g, neighbor))
                open_set.add(neighbor)

    return None, float("inf"), explored  # No path found


def visualize_astar(nodes, edges, path, explored, start, goal, cost, filename="astar_result.png"):
    """Produces a high-quality geographic-style plot of the A* result."""
    G = nx.Graph()
    for name in nodes:
        G.add_node(name)
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)

    pos = {name: (coord[1], coord[0]) for name, coord in nodes.items()}  # lon, lat

    fig, ax = plt.subplots(figsize=(16, 13), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    explored_names = [e[0] for e in explored]
    path_set = set(path)
    path_edges = list(zip(path[:-1], path[1:]))

    # Draw all edges
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#1E3A5F", width=1.2, alpha=0.6)

    # Draw path edges
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                           edge_color="#00D4FF", width=4.0, alpha=0.95)

    # Color nodes
    node_colors = []
    node_sizes  = []
    for n in G.nodes():
        if n == start:
            node_colors.append("#00FF88"); node_sizes.append(500)
        elif n == goal:
            node_colors.append("#FF4B6E"); node_sizes.append(500)
        elif n in path_set:
            node_colors.append("#00D4FF"); node_sizes.append(300)
        elif n in explored_names:
            node_colors.append("#7C5CBF"); node_sizes.append(200)
        else:
            node_colors.append("#2A4A6B"); node_sizes.append(180)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.95)

    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_color="#E8F4FD", font_size=7.5,
                            font_family="monospace",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#0D1117", alpha=0.7, ec="none"))

    # Edge weight labels on path
    edge_labels = {(a, b): f"{w:.1f}km"
                   for a, b, w in edges
                   if (a, b) in [tuple(e) for e in path_edges]
                   or (b, a) in [tuple(e) for e in path_edges]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_color="#00D4FF", font_size=7,
                                 bbox=dict(boxstyle="round", fc="#0D1117", alpha=0.8, ec="none"))

    # Legend
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#00FF88", markersize=10, label=f"Start: {start}"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#FF4B6E", markersize=10, label=f"Goal: {goal}"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#00D4FF", markersize=8,  label="Optimal Path"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#7C5CBF", markersize=8,  label="Explored Nodes"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#2A4A6B", markersize=8,  label="Unexplored"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              facecolor="#131A22", edgecolor="#1E3A5F",
              labelcolor="#E8F4FD", fontsize=9, framealpha=0.9)

    # Stats box
    stats = (
        f"  Algorithm  : A* Search\n"
        f"  Route      : {start} → {goal}\n"
        f"  Optimal Cost: {cost:.2f} km\n"
        f"  Path Nodes : {len(path)}\n"
        f"  Nodes Explored: {len(explored)}"
    )
    ax.text(0.01, 0.99, stats, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            color="#E8F4FD",
            bbox=dict(boxstyle="round", facecolor="#131A22", alpha=0.9, edgecolor="#00D4FF", linewidth=1.2))

    ax.set_title("Karachi City Graph  ─  A* Optimal Vehicle Routing",
                 color="#E8F4FD", fontsize=14, pad=14, fontfamily="monospace")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"✓  A* visualisation saved → {filename}")


def run_astar_task(src=None, dst=None):
    print("\n" + "═"*60)
    print("  TASK 1 ─ A* Informed Search  (Vehicle Routing)")
    print("═"*60)

    graph = build_graph(KARACHI_NODES, KARACHI_EDGES)
    node_names = list(KARACHI_NODES.keys())

    if src is None:
        src = "Clifton"
    if dst is None:
        dst = "Gulshan"

    if src not in KARACHI_NODES or dst not in KARACHI_NODES:
        print(f"  ✗  Unknown node. Valid nodes:\n  {node_names}")
        return

    print(f"\n  Start  : {src}")
    print(f"  Goal   : {dst}")
    print(f"  Heuristic: Euclidean (straight-line) distance")

    t0 = time.perf_counter()
    path, cost, explored = astar(graph, KARACHI_NODES, src, dst)
    elapsed = (time.perf_counter() - t0) * 1000

    if path is None:
        print("  ✗  No path found.")
        return

    print(f"\n  Optimal Path ({cost:.2f} km):")
    print("  " + "  →  ".join(path))
    print(f"\n  Nodes Explored : {len(explored)}")
    print(f"  Total Nodes    : {len(KARACHI_NODES)}")
    print(f"  Search Time    : {elapsed:.3f} ms")

    # Step-by-step exploration table
    print(f"\n  Exploration Order:")
    print(f"  {'#':>3}  {'Node':<20}  {'g(n) km':>9}")
    print(f"  {'─'*3}  {'─'*20}  {'─'*9}")
    for i, (node, g) in enumerate(explored, 1):
        marker = " ◀ GOAL" if node == dst else ""
        print(f"  {i:>3}  {node:<20}  {g:>9.2f}{marker}")

    visualize_astar(KARACHI_NODES, KARACHI_EDGES, path, explored, src, dst, cost)

    # Export path as JSON for the web dashboard
    result = {
        "algorithm": "A*",
        "start": src, "goal": dst,
        "path": path, "cost_km": round(cost, 2),
        "nodes_explored": len(explored),
        "exploration_order": [{"node": n, "g_km": round(g, 2)} for n, g in explored],
        "all_nodes": {n: {"lat": c[0], "lon": c[1]} for n, c in KARACHI_NODES.items()},
        "edges": [{"from": a, "to": b, "km": w} for a, b, w in KARACHI_EDGES],
    }
    with open("astar_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("✓  A* data exported → astar_result.json")
    return result


# ════════════════════════════════════════════════════════════════════════════
# TASK 2 ─ Dynamic Pricing via Random Forest Regression
# ════════════════════════════════════════════════════════════════════════════

def run_pricing_task():
    print("\n" + "═"*60)
    print("  TASK 2 ─ Dynamic Pricing (ML Regression)")
    print("═"*60)

    # ── Load / Generate Dataset ─────────────────────────────────────────
    try:
        df = pd.read_csv("rental_pricing_dataset.csv")
        print(f"\n  Dataset loaded: {len(df):,} records, {len(df.columns)} features")
    except FileNotFoundError:
        print("  Dataset not found — generating now…")
        import subprocess, sys
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        df = pd.read_csv("rental_pricing_dataset.csv")

    print(f"\n  Feature overview:")
    print(df.describe(include="all").loc[["count","mean","std","min","max"]].T.to_string())

    # ── Feature Engineering ─────────────────────────────────────────────
    CATEGORICAL = ["vehicle_category", "day_of_week", "weather", "season"]
    NUMERICAL   = ["month", "is_weekend", "is_holiday",
                   "competitor_base_price", "utilization_rate", "rental_duration_days"]
    TARGET      = "daily_rental_price"

    X = df[CATEGORICAL + NUMERICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n  Train : {len(X_train):,} rows")
    print(f"  Test  : {len(X_test):,} rows")

    # ── Pipeline ────────────────────────────────────────────────────────
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERICAL),
    ])

    models = {
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12,
                                                    min_samples_split=4, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                        max_depth=5, random_state=42),
    }

    results_summary = {}
    best_model_name, best_rmse, best_pipeline = None, float("inf"), None

    for name, model in models.items():
        print(f"\n  ── Training: {name} ──")
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = pipe.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = mean_squared_error(y_test, y_pred) ** 0.5
        r2     = r2_score(y_test, y_pred)

        # 5-fold CV on training set
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        cv_rmse   = (-cv_scores.mean()) ** 0.5

        print(f"    MAE  : ${mae:.2f}")
        print(f"    RMSE : ${rmse:.2f}   (CV RMSE: ${cv_rmse:.2f})")
        print(f"    R²   : {r2:.4f}")
        print(f"    Time : {train_time:.2f}s")

        results_summary[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_RMSE": cv_rmse}

        if rmse < best_rmse:
            best_rmse, best_model_name, best_pipeline = rmse, name, pipe

    print(f"\n  ✓  Best model: {best_model_name} (RMSE ${best_rmse:.2f})")

    # Save model
    with open("pricing_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    print("  ✓  Model saved → pricing_model.pkl")

    # ── Feature Importance ──────────────────────────────────────────────
    inner_model = best_pipeline.named_steps["model"]
    feature_names_cat = best_pipeline.named_steps["prep"]\
                            .named_transformers_["cat"]\
                            .get_feature_names_out(CATEGORICAL).tolist()
    all_features = feature_names_cat + NUMERICAL
    importances  = inner_model.feature_importances_

    imp_df = pd.DataFrame({"feature": all_features, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor="#0D1117")
    for ax in axes:
        ax.set_facecolor("#131A22")

    # Plot 1 – Actual vs Predicted
    ax1 = axes[0]
    y_pred_best = best_pipeline.predict(X_test)
    ax1.scatter(y_test, y_pred_best, alpha=0.35, s=18, color="#00D4FF", edgecolors="none")
    lo, hi = y_test.min(), y_test.max()
    ax1.plot([lo, hi], [lo, hi], color="#FF4B6E", linewidth=1.5, linestyle="--", label="Perfect fit")
    ax1.set_xlabel("Actual Price ($)", color="#A0B4C8"); ax1.set_ylabel("Predicted Price ($)", color="#A0B4C8")
    ax1.set_title(f"{best_model_name}\nActual vs Predicted", color="#E8F4FD", fontsize=11)
    ax1.legend(fontsize=8, facecolor="#0D1117", labelcolor="#E8F4FD")
    ax1.tick_params(colors="#6B8299"); ax1.text(0.05, 0.92, f"R² = {results_summary[best_model_name]['R2']:.4f}",
        transform=ax1.transAxes, color="#00FF88", fontsize=10, fontfamily="monospace")

    # Plot 2 – Feature importance
    ax2 = axes[1]
    colors_bar = ["#00D4FF" if i < 5 else "#7C5CBF" for i in range(len(imp_df))]
    bars = ax2.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors_bar[::-1])
    ax2.set_xlabel("Importance", color="#A0B4C8")
    ax2.set_title("Top Feature Importances", color="#E8F4FD", fontsize=11)
    ax2.tick_params(colors="#6B8299", axis="x")
    ax2.tick_params(colors="#E8F4FD", axis="y", labelsize=8)

    # Plot 3 – Model comparison
    ax3 = axes[2]
    model_names = list(results_summary.keys())
    metrics_vals = [results_summary[m] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.28
    b1 = ax3.bar(x - width, [m["MAE"] for m in metrics_vals],   width, label="MAE",     color="#00D4FF")
    b2 = ax3.bar(x,         [m["RMSE"] for m in metrics_vals],  width, label="RMSE",    color="#7C5CBF")
    b3 = ax3.bar(x + width, [m["CV_RMSE"] for m in metrics_vals], width, label="CV RMSE", color="#FF4B6E")
    ax3.set_xticks(x); ax3.set_xticklabels(model_names, color="#E8F4FD", fontsize=9)
    ax3.set_ylabel("Error ($)", color="#A0B4C8")
    ax3.set_title("Model Comparison", color="#E8F4FD", fontsize=11)
    ax3.legend(facecolor="#0D1117", labelcolor="#E8F4FD", fontsize=8)
    ax3.tick_params(colors="#6B8299")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E3A5F")

    fig.suptitle("Dynamic Pricing Model  ─  Evaluation Dashboard",
                 color="#E8F4FD", fontsize=14, y=1.01, fontfamily="monospace")
    plt.tight_layout()
    plt.savefig("pricing_results.png", dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("  ✓  Plots saved → pricing_results.png")

    # ── Sample Predictions ───────────────────────────────────────────────
    print("\n  Sample Predictions:")
    samples = pd.DataFrame([
        {"vehicle_category":"Economy",  "day_of_week":"Friday",   "weather":"Sunny",  "season":"Summer",
         "month":7, "is_weekend":1, "is_holiday":0, "competitor_base_price":32, "utilization_rate":82, "rental_duration_days":2},
        {"vehicle_category":"Luxury",   "day_of_week":"Monday",   "weather":"Rainy",  "season":"Winter",
         "month":1, "is_weekend":0, "is_holiday":1, "competitor_base_price":130,"utilization_rate":55, "rental_duration_days":1},
        {"vehicle_category":"SUV",      "day_of_week":"Saturday", "weather":"Snowy",  "season":"Winter",
         "month":12,"is_weekend":1, "is_holiday":0, "competitor_base_price":88, "utilization_rate":91, "rental_duration_days":3},
        {"vehicle_category":"Compact",  "day_of_week":"Wednesday","weather":"Cloudy", "season":"Spring",
         "month":4, "is_weekend":0, "is_holiday":0, "competitor_base_price":42, "utilization_rate":60, "rental_duration_days":7},
    ])
    preds = best_pipeline.predict(samples)
    print(f"\n  {'Vehicle':<12}  {'Day':<12}  {'Weather':<8}  {'Predicted Price':>15}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*15}")
    for i, row in samples.iterrows():
        print(f"  {row['vehicle_category']:<12}  {row['day_of_week']:<12}  {row['weather']:<8}  ${preds[i]:>14.2f}")

    # Export evaluation JSON
    eval_data = {
        "best_model": best_model_name,
        "metrics": {k: {m: round(v, 4) for m, v in vs.items()} for k, vs in results_summary.items()},
        "feature_importances": imp_df[["feature","importance"]].assign(
            importance=imp_df["importance"].round(5)).to_dict(orient="records"),
        "sample_predictions": [
            {**samples.iloc[i].to_dict(), "predicted_price": round(float(preds[i]), 2)}
            for i in range(len(samples))
        ],
    }
    with open("pricing_results.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    print("  ✓  Evaluation exported → pricing_results.json")
    return eval_data


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Intelligent Vehicle Routing & Dynamic Pricing System")
    parser.add_argument("--task", choices=["astar", "pricing", "both"], default="both")
    parser.add_argument("--src",  default="Clifton",  help="A* start node")
    parser.add_argument("--dst",  default="Gulshan",  help="A* goal node")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Intelligent Vehicle Routing & Dynamic Pricing System   ║")
    print("║  FAST-NUCES AI/ML Project  ─  2026                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.task in ("astar", "both"):
        run_astar_task(args.src, args.dst)

    if args.task in ("pricing", "both"):
        run_pricing_task()

    print("\n✓  All tasks complete.\n")


if __name__ == "__main__":
    main()
