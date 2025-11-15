import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("global_air_quality_data_10000.csv")

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Drop missing values for key columns
df = df.dropna(subset=['City', 'Country', 'PM2.5', 'Temperature', 'Humidity'])

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Basic EDA
print("Data Preview:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# ---------- VISUALIZATIONS ----------

# Air quality across countries
plt.figure(figsize=(12,6))
country_pm25 = df.groupby('Country')['PM2.5'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=country_pm25.values, y=country_pm25.index, palette='viridis')
plt.title("Top 10 Countries by Average PM2.5 Levels")
plt.xlabel("Average PM2.5")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("pm25_by_country.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[['PM2.5', 'Temperature', 'Humidity', 'Wind Speed']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Pollution and Weather")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# PM2.5 Trends Over Time (aggregated)
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_pm25 = df.groupby('YearMonth')['PM2.5'].mean()
monthly_pm25.plot(figsize=(12,5), title='Global Average PM2.5 Over Time')
plt.ylabel("Average PM2.5")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig("pm25_trend_over_time.png")
plt.close()

# ---------- MACHINE LEARNING ----------

# Predict PM2.5 from weather conditions
features = ['Temperature', 'Humidity', 'Wind Speed']
target = 'PM2.5'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Results ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
plt.figure(figsize=(8,5))
sns.barplot(x=model.feature_importances_, y=features, palette="magma")
plt.title("Feature Importance for PM2.5 Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()