"""Exploratory analysis and a PM2.5 regression on the Global Air Quality dataset.

Run:  python air_quality_insights.py
Charts are written to ./images/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; works on machines without a display

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).parent / "global_air_quality_data_10000.csv"
OUT_DIR = Path(__file__).parent / "images"
WEATHER = ["Temperature", "Humidity", "Wind Speed"]
TARGET = "PM2.5"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=["City", "Country", TARGET, *WEATHER])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def save(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / fig_name, dpi=120)
    plt.close()


def plot_eda(df: pd.DataFrame) -> None:
    # Average PM2.5 by country
    plt.figure(figsize=(12, 6))
    country_pm25 = df.groupby("Country")[TARGET].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=country_pm25.values, y=country_pm25.index, hue=country_pm25.index,
                palette="viridis", legend=False)
    plt.title("Top 10 Countries by Average PM2.5")
    plt.xlabel("Average PM2.5 (µg/m³)")
    plt.ylabel("Country")
    save("pm25_by_country.png")

    # Correlation between PM2.5 and weather
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[[TARGET, *WEATHER]].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1)
    plt.title("Correlation Between PM2.5 and Weather")
    save("correlation_heatmap.png")

    # Monthly global average
    monthly = df.groupby(df["Date"].dt.to_period("M"))[TARGET].mean()
    monthly.plot(figsize=(12, 5), title="Global Average PM2.5 by Month", marker="o")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.xlabel("Month")
    plt.grid(True)
    save("pm25_trend_over_time.png")


def train_model(df: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        df[WEATHER], df[TARGET], test_size=0.2, random_state=42
    )

    # Baseline: always predict the training-set mean. A real model has to beat this.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    print("\n--- PM2.5 from weather: test-set results ---")
    for name, m in [("Mean baseline", baseline), ("Random forest", model)]:
        pred = m.predict(X_test)
        print(f"{name:<14}  MSE = {mean_squared_error(y_test, pred):8.2f}   R² = {r2_score(y_test, pred):5.2f}")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=model.feature_importances_, y=WEATHER, hue=WEATHER, palette="magma", legend=False)
    plt.title("Random Forest Feature Importance for PM2.5")
    save("feature_importance.png")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data(DATA_FILE)

    print(f"Loaded {len(df):,} rows covering {df['City'].nunique()} cities "
          f"in {df['Country'].nunique()} countries ({df['Date'].min():%b %Y} – {df['Date'].max():%b %Y}).")
    print("\nSummary statistics:")
    print(df.describe().round(2))

    plot_eda(df)
    train_model(df)
    print(f"\nCharts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
