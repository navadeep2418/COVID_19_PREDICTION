# =============================================================================
#  COVID-19 Daily Case Prediction using LSTM Deep Learning
#  Real-Life Incident: The 2020-2022 Global COVID-19 Pandemic
#
#  Dataset  : Johns Hopkins University CSSE COVID-19 time-series (public)
#  Model    : LSTM (Long Short-Term Memory) Neural Network
#  Goal     : Predict next-day confirmed cases from historical sequences
#
#  Run in VS Code:
#    1. pip install -r requirements.txt
#    2. python covid19_prediction.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings, os, requests
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  COVID-19 Daily Case Prediction — LSTM Deep Learning Project")
print("=" * 65)
print(f"  TensorFlow : {tf.__version__}")
print(f"  NumPy      : {np.__version__}")
print(f"  Pandas     : {pd.__version__}")
print()

# =============================================================================
# STEP 1 — LOAD REAL COVID-19 DATA
# =============================================================================
print("[1/7] Loading real COVID-19 data from Johns Hopkins CSSE ...")

URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

try:
    df_raw = pd.read_csv(URL)
    print("      Data loaded from GitHub (live).")
except Exception:
    print("      Network unavailable — generating realistic synthetic dataset.")
    # Realistic synthetic data mimicking India's COVID waves
    dates = pd.date_range("2020-03-01", "2022-12-31", freq="D")
    n = len(dates)
    t = np.arange(n)
    # Wave 1 (mid-2020), Wave 2 (Apr 2021 peak), Wave 3 (Jan 2022 Omicron)
    wave1 = 90_000  * np.exp(-((t - 120)**2) / (2 * 40**2))
    wave2 = 400_000 * np.exp(-((t - 430)**2) / (2 * 35**2))
    wave3 = 300_000 * np.exp(-((t - 670)**2) / (2 * 30**2))
    noise = np.random.normal(0, 8000, n)
    cases = np.clip(wave1 + wave2 + wave3 + noise, 0, None).astype(int)
    df_raw = pd.DataFrame({"date": dates, "new_cases": cases})
    df_raw.to_csv("outputs/synthetic_covid_india.csv", index=False)
    # Reshape to match JHU format
    df_raw = None  # signal synthetic branch below

# ── Parse JHU cumulative → daily new cases for India ─────────────────────
if df_raw is not None:
    india = df_raw[df_raw["Country/Region"] == "India"].copy()
    india = india.drop(columns=["Province/State", "Country/Region", "Lat", "Long"])
    india = india.sum(axis=0)  # sum provinces
    india.index = pd.to_datetime(india.index, format="%m/%d/%y")
    india = india.sort_index()
    daily = india.diff().fillna(0).clip(lower=0)
    df = pd.DataFrame({"date": daily.index, "new_cases": daily.values})
    df = df[(df["date"] >= "2020-03-01") & (df["date"] <= "2022-12-31")]
    df.reset_index(drop=True, inplace=True)
else:
    df = pd.read_csv("outputs/synthetic_covid_india.csv", parse_dates=["date"])

print(f"      Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"      Total days : {len(df)}")
print(f"      Peak cases : {df['new_cases'].max():,.0f}")
print(f"      Avg cases  : {df['new_cases'].mean():,.0f}")

# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[2/7] Generating EDA plots ...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "COVID-19 India — Exploratory Data Analysis\n"
    "(Johns Hopkins CSSE Data | 2020–2022)",
    fontsize=14, fontweight="bold"
)

# 2a. Raw daily cases timeline
ax = axes[0, 0]
ax.fill_between(df["date"], df["new_cases"], alpha=0.4, color="#e74c3c")
ax.plot(df["date"], df["new_cases"], color="#c0392b", linewidth=0.8)
ax.set_title("Daily New COVID-19 Cases — India", fontweight="bold")
ax.set_ylabel("New Cases")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
# Mark waves
for wave_date, label, color in [
    ("2020-09-16", "Wave 1 Peak\n~97K cases",   "#8e44ad"),
    ("2021-05-06", "Wave 2 Peak\n~414K cases",  "#2980b9"),
    ("2022-01-20", "Wave 3 Peak\n~347K cases",  "#27ae60"),
]:
    wd = pd.Timestamp(wave_date)
    if wd in df["date"].values:
        val = df.loc[df["date"] == wd, "new_cases"].values[0]
        ax.annotate(label, xy=(wd, val), xytext=(wd, val * 0.7),
                    arrowprops=dict(arrowstyle="->", color=color),
                    fontsize=8, color=color, ha="center")
ax.grid(True, alpha=0.3)

# 2b. 7-day rolling average
ax = axes[0, 1]
rolling7  = df["new_cases"].rolling(7).mean()
rolling30 = df["new_cases"].rolling(30).mean()
ax.plot(df["date"], df["new_cases"],  alpha=0.3, color="gray",    linewidth=0.6, label="Daily")
ax.plot(df["date"], rolling7,         color="#e74c3c", linewidth=1.5, label="7-day MA")
ax.plot(df["date"], rolling30,        color="#2980b9", linewidth=2,   label="30-day MA")
ax.set_title("Rolling Averages (7-day & 30-day)", fontweight="bold")
ax.set_ylabel("Cases")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True, alpha=0.3)

# 2c. Monthly total cases heatmap by year
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
pivot = df.pivot_table(values="new_cases", index="year", columns="month", aggfunc="sum")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
sns.heatmap(pivot, ax=axes[1, 0], cmap="YlOrRd", fmt=".0f",
            annot=True, annot_kws={"size": 7}, linewidths=0.5,
            cbar_kws={"label": "Total Cases"})
axes[1, 0].set_title("Monthly Total Cases Heatmap (by Year)", fontweight="bold")
axes[1, 0].set_ylabel("Year")

# 2d. Distribution of daily cases
ax = axes[1, 1]
ax.hist(df["new_cases"], bins=60, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(df["new_cases"].mean(),   color="red",    linestyle="--", label=f'Mean  = {df["new_cases"].mean():,.0f}')
ax.axvline(df["new_cases"].median(), color="orange", linestyle="--", label=f'Median= {df["new_cases"].median():,.0f}')
ax.set_title("Distribution of Daily Case Counts", fontweight="bold")
ax.set_xlabel("Daily New Cases")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/eda_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("      Saved: outputs/eda_analysis.png")

# =============================================================================
# STEP 3 — PREPROCESSING & SEQUENCE CREATION
# =============================================================================
print("\n[3/7] Preprocessing data ...")

# Normalize to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[["new_cases"]])

LOOK_BACK = 30   # use last 30 days to predict next day
TRAIN_RATIO = 0.80

def create_sequences(data, look_back=30):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, LOOK_BACK)
X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, timesteps, features)

split = int(len(X) * TRAIN_RATIO)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"      Look-back window : {LOOK_BACK} days")
print(f"      Training samples : {X_train.shape[0]}")
print(f"      Test samples     : {X_test.shape[0]}")
print(f"      Input shape      : {X_train.shape}")

# =============================================================================
# STEP 4 — BUILD LSTM MODEL
# =============================================================================
print("\n[4/7] Building LSTM model ...")

def build_lstm(look_back=30):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(look_back, 1)),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ], name="COVID_LSTM")
    return model

model = build_lstm(LOOK_BACK)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
model.summary()

total_params = model.count_params()
print(f"\n      Total trainable parameters: {total_params:,}")

# =============================================================================
# STEP 5 — TRAIN MODEL
# =============================================================================
print("\n[5/7] Training model ...")

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=15,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=7, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint("outputs/best_model.keras",
                              monitor="val_loss", save_best_only=True, verbose=0),
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=cb_list,
    verbose=1
)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LSTM Training History", fontsize=13, fontweight="bold")

ax1.plot(history.history["loss"],     label="Train Loss", linewidth=2)
ax1.plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
ax1.set_title("MSE Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history["mae"],     label="Train MAE", linewidth=2, color="orange")
ax2.plot(history.history["val_mae"], label="Val MAE",   linewidth=2, linestyle="--", color="red")
ax2.set_title("Mean Absolute Error")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print("      Saved: outputs/training_history.png")

# =============================================================================
# STEP 6 — EVALUATE & VISUALIZE PREDICTIONS
# =============================================================================
print("\n[6/7] Evaluating model ...")

# Inverse-transform predictions
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │          Model Evaluation Metrics       │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │  RMSE : {rmse:>15,.2f} cases           │")
print(f"  │  MAE  : {mae:>15,.2f} cases           │")
print(f"  │  MAPE : {mape:>15.2f} %              │")
print(f"  │  R²   : {r2:>15.4f}                  │")
print(f"  └─────────────────────────────────────────┘")

# Prediction dates
test_dates = df["date"].iloc[LOOK_BACK + split:].reset_index(drop=True)
if len(test_dates) < len(y_true):
    test_dates = pd.date_range(df["date"].iloc[-len(y_true)], periods=len(y_true))

# Plot predictions
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle(
    "COVID-19 Daily Cases — LSTM Prediction Results\n"
    "Real-Life Data: India 2020–2022",
    fontsize=13, fontweight="bold"
)

# Full timeline
ax = axes[0]
train_vals = scaler.inverse_transform(scaled[:split + LOOK_BACK]).flatten()
train_dates = df["date"].iloc[:split + LOOK_BACK]
ax.plot(train_dates, train_vals, color="#2ecc71", linewidth=1.2, label="Training Data", alpha=0.7)
ax.plot(test_dates[:len(y_true)], y_true, color="#3498db", linewidth=1.5, label="Actual (Test Period)")
ax.plot(test_dates[:len(y_pred)], y_pred, color="#e74c3c", linewidth=1.5,
        linestyle="--", label="LSTM Prediction")
ax.set_title("Full Timeline: Training Data + Test Predictions", fontweight="bold")
ax.set_ylabel("Daily New Cases")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, f"R² = {r2:.4f}  |  RMSE = {rmse:,.0f}  |  MAE = {mae:,.0f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# Actual vs Predicted zoom
ax = axes[1]
n_show = min(180, len(y_true))
ax.plot(test_dates[:n_show], y_true[:n_show], color="#2980b9",
        linewidth=2, label="Actual Cases", marker="o", markersize=2)
ax.plot(test_dates[:n_show], y_pred[:n_show], color="#e74c3c",
        linewidth=2, linestyle="--", label="Predicted Cases", marker="s", markersize=2)
ax.fill_between(test_dates[:n_show], y_true[:n_show], y_pred[:n_show],
                alpha=0.15, color="purple", label="Prediction Error")
ax.set_title(f"Zoomed View — Last {n_show} Days of Test Set", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily New Cases")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("      Saved: outputs/predictions.png")

# Scatter: Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_true, y_pred, alpha=0.5, color="#3498db", s=15, label="Predictions")
lim = max(y_true.max(), y_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Cases")
ax.set_ylabel("Predicted Cases")
ax.set_title(f"Actual vs Predicted  (R² = {r2:.4f})", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("      Saved: outputs/scatter_actual_vs_predicted.png")

# =============================================================================
# STEP 7 — FUTURE 30-DAY FORECAST
# =============================================================================
print("\n[7/7] Generating 30-day future forecast ...")

def forecast_future(model, last_sequence, scaler, days=30):
    predictions = []
    seq = last_sequence.copy()
    for _ in range(days):
        inp = seq.reshape(1, LOOK_BACK, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        predictions.append(pred)
        seq = np.roll(seq, -1)
        seq[-1] = pred
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

last_seq   = scaled[-LOOK_BACK:, 0]
future     = forecast_future(model, last_seq, scaler, days=30)
future_dates = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=30)

fig, ax = plt.subplots(figsize=(14, 6))
# Show last 90 days of actual data
ax.plot(df["date"].iloc[-90:], df["new_cases"].iloc[-90:],
        color="#2980b9", linewidth=2, label="Historical (Last 90 Days)")
ax.plot(future_dates, future, color="#e74c3c", linewidth=2.5,
        linestyle="--", marker="o", markersize=5, label="30-Day Forecast")
ax.fill_between(future_dates,
                future * 0.80, future * 1.20,
                alpha=0.15, color="#e74c3c", label="±20% Confidence Band")
ax.axvline(df["date"].iloc[-1], color="gray", linestyle=":", linewidth=1.5, label="Forecast Start")
ax.set_title("30-Day COVID-19 Case Forecast — India", fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Predicted Daily Cases")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/future_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
print("      Saved: outputs/future_forecast.png")

# Save forecast to CSV
forecast_df = pd.DataFrame({"date": future_dates, "predicted_cases": future.astype(int)})
forecast_df.to_csv("outputs/30day_forecast.csv", index=False)
print("      Saved: outputs/30day_forecast.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 65)
print("  PROJECT COMPLETE — All outputs saved in ./outputs/")
print("=" * 65)
print(f"  Files generated:")
print(f"    outputs/eda_analysis.png")
print(f"    outputs/training_history.png")
print(f"    outputs/predictions.png")
print(f"    outputs/scatter_actual_vs_predicted.png")
print(f"    outputs/future_forecast.png")
print(f"    outputs/30day_forecast.csv")
print(f"    outputs/best_model.keras")
print()
print(f"  Final Metrics:")
print(f"    RMSE  = {rmse:>12,.2f}")
print(f"    MAE   = {mae:>12,.2f}")
print(f"    MAPE  = {mape:>11.2f}%")
print(f"    R²    = {r2:>12.4f}")
print("=" * 65)
