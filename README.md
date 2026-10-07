# California House Price Prediction

A regression project predicting median house values across California districts using income, location, and housing characteristics.

## Dataset
California Housing dataset (~20,640 districts) — includes location, income, housing age, room counts, and population statistics.

## Exploratory Data Analysis — Key Findings

- **Median Income** was the strongest individual predictor (correlation 0.688) — wealth of an area drives housing prices far more than any other single feature.
- **Location** matters significantly, but not as a simple straight-line relationship — longitude and latitude individually showed almost no correlation (-0.046), but combined, they revealed a strong price cluster in the San Francisco Bay Area.
- **Ocean Proximity** showed a clear price gradient: INLAND districts averaged $124,805, roughly half the price of coastal categories (NEAR BAY, NEAR OCEAN, <1H OCEAN), which clustered around $240,000–260,000. ISLAND showed the highest average ($380,440) but was based on only 5 districts out of 20,000+, making it statistically unreliable.
- **Raw counts** (total_rooms, total_bedrooms, population, households) showed almost no individual correlation with price. Engineering ratios improved this — `bedrooms_per_room` (-0.256) was a notably stronger predictor than any raw count, since a higher bedroom-to-room ratio indicates smaller, more basic housing.

## Feature Engineering
- `rooms_per_household` = total_rooms / households
- `bedrooms_per_room` = total_bedrooms / total_rooms
- Dropped raw columns (total_rooms, total_bedrooms, population, households) after engineering ratios, since the ratios carried more signal than the raw counts

## Data Preprocessing
- Filled missing `total_bedrooms` values (~1% missing) with median, then recalculated `bedrooms_per_room` using the filled values
- One-hot encoded `ocean_proximity` (no inherent order between categories)

## Models Compared (5-fold Cross-Validation, R²)
Multiple regression algorithms were compared using cross-validation, including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost. XGBoost performed best and was selected for hyperparameter tuning.

**Best parameters:** `learning_rate=0.1`, `max_depth=7`, `n_estimators=300`
**Tuned CV R²:** 0.834

## Final Test Set Results
- **R²:** 0.817 (model explains ~82% of variance in house prices)
- **MAE:** $31,497
- **RMSE:** $49,034

MAE of $31,497 represents roughly 15% of the average house value ($205,500), indicating reasonably accurate predictions given the available features — real-world housing prices depend on many factors (property condition, school district, local amenities) not captured in this dataset.

## What I'd Improve Next
- Try additional feature engineering (e.g., distance to nearest major city)
- Address the dataset's known price-cap artifact (values capped at $500,001)
- Experiment with log-transforming the target, since house prices are typically right-skewed
- Explore stacking multiple models to combine their strengths

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
