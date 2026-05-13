import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_PATH      = os.path.join(MODELS_DIR, "xgb_rul_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
FI_PLOT_PATH    = os.path.join(MODELS_DIR, "feature_importance.png")
SCATTER_PATH    = os.path.join(MODELS_DIR, "pred_vs_actual.png")
METRICS_PATH    = os.path.join(MODELS_DIR, "metrics.csv")

NON_FEATURE_COLS = {"unit", "cycle", "RUL"}

XGB_PARAMS = {
    "n_estimators":     800,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}


def load_features():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    return train, test


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_xy(df):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["RUL"].values
    return X, y, feature_cols


def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    y_pred   = np.clip(y_pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    return y_pred, rmse, mae, r2


def plot_feature_importance(model, feature_cols):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(
        fi_df["feature"][::-1],
        fi_df["importance"][::-1],
        color="#f97316",
        edgecolor="none",
        height=0.65
    )

    ax.set_xlabel("Importance Score", color="#e2e8f0", fontsize=11)
    ax.set_title("Top 20 Feature Importances — XGBoost RUL Model", color="#f97316", fontsize=13, pad=14)
    ax.tick_params(colors="#e2e8f0", labelsize=9)
    ax.spines[:].set_color("#334155")
    ax.xaxis.label.set_color("#e2e8f0")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {FI_PLOT_PATH}")


def plot_pred_vs_actual(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.scatter(y_test, y_pred, alpha=0.35, s=14, color="#f97316", edgecolors="none", label="Predictions")

    lims = [0, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, color="#94a3b8", linewidth=1.2, linestyle="--", label="Perfect fit")

    ax.set_xlabel("Actual RUL",    color="#e2e8f0", fontsize=11)
    ax.set_ylabel("Predicted RUL", color="#e2e8f0", fontsize=11)
    ax.set_title("Predicted vs Actual RUL", color="#f97316", fontsize=13, pad=14)
    ax.tick_params(colors="#e2e8f0", labelsize=9)
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#16213e", labelcolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    plt.savefig(SCATTER_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {SCATTER_PATH}")


def save_metrics(rmse, mae, r2):
    df = pd.DataFrame([{"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}])
    df.to_csv(METRICS_PATH, index=False)
    print(f"Saved: {METRICS_PATH}")


def run():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading engineered features...")
    train, test = load_features()

    X_train, y_train, feature_cols = prepare_xy(train)
    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
    X_test,  y_test,  _            = prepare_xy(test_last)

    print(f"  Train samples : {X_train.shape[0]:,}  |  Features: {X_train.shape[1]}")
    print(f"  Test  samples : {X_test.shape[0]:,}")

    print("Training XGBoost regressor...")
    model, scaler = train_model(X_train, y_train)
    print("  Training complete.")

    print("Evaluating on test set...")
    y_pred, rmse, mae, r2 = evaluate(model, scaler, X_test, y_test)

    print("\n" + "=" * 40)
    print("  MODEL PERFORMANCE")
    print("=" * 40)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print("=" * 40 + "\n")

    print("Generating plots...")
    plot_feature_importance(model, feature_cols)
    plot_pred_vs_actual(y_test, y_pred)

    print("Saving model and scaler...")
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")

    save_metrics(rmse, mae, r2)

    print("✅")
    return model, scaler, feature_cols, y_test, y_pred, rmse, mae, r2


if __name__ == "__main__":
    run()