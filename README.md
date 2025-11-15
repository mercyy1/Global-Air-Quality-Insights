## Why This Project Exists

Air pollution is one of the most pressing global health issues, yet the data around it is often scattered or underutilized. I wanted to create something that made them meaningful. This project explores global air quality data from a large open dataset, then breaks it down through visualization and light predictive modeling to help uncover trends in pollution levels, temperature, humidity, and more across different cities.

---

## What It Does

* Cleans and processes a large dataset of air quality metrics
* Visualizes pollution trends across major pollutants (PM2.5, NO2, O3, etc.)
* Looks at how air quality relates to environmental conditions like humidity and temperature
* Builds a basic regression model to predict PM2.5 levels based on other environmental factors

---

## Tools Used

* Python (Pandas, Matplotlib, Seaborn, Scikit-learn)
* Jupyter Notebook / script mode
* CSV data from [Kaggle](https://www.kaggle.com/datasets/waqi786/global-air-quality-dataset)

---

## How to Run It

1. Clone this repo:

   ```
   git clone https://github.com/your-username/urban-air-quality-insights.git
   cd urban-air-quality-insights
   ```

2. Make sure you have the following Python libraries installed:

   ```
   pip install pandas matplotlib seaborn scikit-learn
   ```

3. Place the dataset CSV file (`global_air_quality_data_10000.csv`) in the root directory.

4. Run the main script:

   ```
   python air_quality_insights.py
   ```

---

## What You’ll See

* Cleaned summary of the dataset
* Charts showing top polluted cities and pollutant comparisons
* Correlation heatmaps
* A small ML model predicting PM2.5 based on weather and pollutant data
* Simple printed insights about environmental patterns

---

