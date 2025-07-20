import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows where target 'Rating' is missing
df = df.dropna(subset=['Rating'])

# Clean 'Duration' → convert "109 min" to 109 (integer)
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)

# Clean 'Year' → remove parentheses
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(float)

# Clean 'Votes' → make numeric
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Drop remaining rows with any missing values
df.dropna(subset=['Duration', 'Year', 'Votes', 'Genre', 'Director'], inplace=True)

# Select features and target
X = df[['Genre', 'Director', 'Duration', 'Year', 'Votes']]
y = df['Rating']

# Define column types
categorical_cols = ['Genre', 'Director']
numerical_cols = ['Duration', 'Year', 'Votes']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Build the model pipeline
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# Plot actual vs predicted ratings
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
