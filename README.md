# Exploring Recipes dataset via Random Forest
```python
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px

pd.options.plotting.backend = 'plotly'

# from lec_utils import * # Feel free to uncomment and use this. It'll make your plotly graphs look like ours in lecture!
```


## Step 1: Introduction

In this project, I aim to **predict the final average rating of recipes** on Food.com based on their features.
Understanding what makes a recipe highly rated can help platforms improve their recommendation systems and guide users toward recipes they are more likely to enjoy. The target for prediction will be the **average rating per recipe**, computed from individual user ratings.



```python
recipes = pd.read_csv("~/downloads/RAW_recipes.csv")
interactions = pd.read_csv("~/downloads/RAW_interactions.csv")

print(f"Number of recipes: {recipes.shape[0]}")
print(f"Number of interactions: {interactions.shape[0]}")

print("\nRecipe columns:")
print(recipes.columns.tolist())

print("\nInteraction columns:")
print(interactions.columns.tolist())

```


## Step 2: Data Cleaning and Exploratory Data Analysis

This section merges the recipes and interactions datasets, replaces 0 ratings with NaN, and computes the average rating per recipe. It then parses the nutrition column into individual components (e.g., calories, protein), converts date columns to datetime format, and filters out extreme values using the 2.5–97.5 percent quantiles for minutes, ingredients, steps, and average rating. Afterward, it generates visualizations to explore rating distributions and trends, and fills missing average ratings with the global mean.


```python
merged = pd.merge(recipes, interactions, how="left", left_on="id", right_on="recipe_id")
merged["rating"] = merged["rating"].replace(0, np.nan)

avg_rating = merged.groupby("id")["rating"].mean()
merged["avg_rating"] = merged["id"].map(avg_rating)
merged["nutrition"] = merged["nutrition"].apply(eval)
nutrition_cols = ["calories", "total_fat", "sugar", "sodium", "protein", "saturated_fat", "carbohydrates"]
for i, col in enumerate(nutrition_cols):
    merged[col] = merged["nutrition"].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)

merged["submitted"] = pd.to_datetime(merged["submitted"])
merged["date"] = pd.to_datetime(merged["date"], errors='coerce')

q_minutes = merged["minutes"].quantile([0.025, 0.975])
q_ingredients = merged["n_ingredients"].quantile([0.025, 0.975])
q_steps = merged["n_steps"].quantile([0.025, 0.975])
q_rating = merged["avg_rating"].quantile([0.025, 0.975])

filtered = merged[
    (merged["minutes"] >= q_minutes[0.025]) & (merged["minutes"] <= q_minutes[0.975]) &
    (merged["n_ingredients"] >= q_ingredients[0.025]) & (merged["n_ingredients"] <= q_ingredients[0.975]) &
    (merged["n_steps"] >= q_steps[0.025]) & (merged["n_steps"] <= q_steps[0.975]) &
    (merged["avg_rating"] >= q_rating[0.025]) & (merged["avg_rating"] <= q_rating[0.975])
].copy()


fig1 = px.histogram(merged, x="avg_rating", nbins=20, title="Distribution of Average Ratings")
fig1.show()

fig2 = px.scatter(filtered, x="minutes", y="avg_rating",
                  title="Average Rating vs. Cooking Time (Filtered)",
                  opacity=0.5, trendline="ols")
fig2.show()

bar_df = filtered.groupby("n_ingredients").agg(
    mean_rating=("avg_rating", "mean"),
    count=("avg_rating", "count")
).reset_index()


line_df = filtered.groupby("n_steps").agg(mean_rating=("avg_rating", "mean")).reset_index()

fig4 = px.line(line_df, x="n_steps", y="mean_rating",
               title="Average Rating Trend by Number of Steps")
fig4.show()

grouped = merged.groupby("n_ingredients")["avg_rating"].agg(["count", "mean"]).reset_index()
grouped.columns = ["n_ingredients", "num_recipes", "mean_avg_rating"]
grouped_sorted = grouped.sort_values(by="n_ingredients")

grouped_sorted.head(10)

merged["avg_rating_filled"] = merged["avg_rating"].fillna(merged["avg_rating"].mean())

```


## Step 3: Framing a Prediction Problem

In this project, I chose to predict the average rating of a recipe based on its known attributes at the time of submission. I framed this as a regression task because the target variable—average user rating—is continuous. I selected this prediction problem because it aligns well with real-world recommendation goals and allows me to explore which features contribute most to perceived recipe quality. I ensured that all predictors are available before user feedback, making the setup both practical and forward-looking.

## Step 4: Baseline Model

This section builds a baseline linear regression model to predict average recipe ratings using original features: minutes, n_steps, n_ingredients, and tags. It splits the data into training and testing sets, extracts the first tag from the list of tags, and applies preprocessing with standardization for numeric features and one-hot encoding for categorical features. The model is trained using a Pipeline, and its performance is evaluated using mean squared error (MSE) and R-squared (R²) on the test set.


```python
model_df = filtered[["minutes", "n_steps", "n_ingredients", "tags", "avg_rating"]].dropna()

X = model_df.drop(columns=["avg_rating"])
y = model_df["avg_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def simplify_tags(tag_str):
    try:
        tags = eval(tag_str)
        return tags[0] if isinstance(tags, list) and tags else "unknown"
    except:
        return "unknown"

X_train["tags"] = X_train["tags"].apply(simplify_tags)
X_test["tags"] = X_test["tags"].apply(simplify_tags)

numeric_features = ["minutes", "n_steps", "n_ingredients"]
categorical_features = ["tags"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Baseline Linear Regression Results (4 original features):")
print(f"  MSE: {mse:.4f}")
print(f"  R^2: {r2:.4f}")



```


## Step 5: Final Model

This section builds a final Random Forest model using four engineered features and the first tag. It applies preprocessing, performs hyperparameter tuning with GridSearchCV, and evaluates the model using MSE and R² on the test set.


```python
model_df = filtered[["minutes", "n_steps", "n_ingredients", "tags", "avg_rating"]].dropna()
model_df["log_minutes"] = np.log1p(model_df["minutes"])
model_df["step_density"] = model_df["n_steps"] / model_df["minutes"].replace(0, np.nan)
model_df["ingredient_density"] = model_df["n_ingredients"] / model_df["minutes"].replace(0, np.nan)
model_df["steps_per_ing"] = model_df["n_steps"] / model_df["n_ingredients"].replace(0, np.nan)

feature_cols = ["log_minutes", "step_density", "ingredient_density", "steps_per_ing", "tags"]
X = model_df[feature_cols]
y = model_df["avg_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def simplify_tags(tag_str):
    try:
        tags = eval(tag_str)
        return tags[0] if isinstance(tags, list) and tags else "unknown"
    except:
        return "unknown"

X_train["tags"] = X_train["tags"].apply(simplify_tags)
X_test["tags"] = X_test["tags"].apply(simplify_tags)

numeric_features = ["log_minutes", "step_density", "ingredient_density", "steps_per_ing"]
categorical_features = ["tags"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [5, 10, 15],
    "regressor__min_samples_split": [2, 5]
}

total_combinations = len(ParameterGrid(param_grid))
print(f"🔧 Total grid combinations: {total_combinations}")

start_time = time.time()

grid_search = GridSearchCV(
    final_pipeline,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)

end_time = time.time()
duration = end_time - start_time
print(f"\n⏱️ Total training time: {duration:.2f} seconds")

y_pred = grid_search.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Final Model: Random Forest with 4 Engineered Features")
print(f"Best Params: {grid_search.best_params_}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

```



```python

```
