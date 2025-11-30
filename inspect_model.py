import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

with open('-Bengaluru-House-Price-Prediction/house_price_model.pkl','rb') as f:
    model = pickle.load(f)

# Load your data
df = pd.read_csv('-Bengaluru-House-Price-Prediction/cleaned_bengaluru_data.csv')

# Create price_per_sqft if it doesn't exist
if 'price_per_sqft' not in df.columns and 'price' in df.columns and 'total_sqft' in df.columns:
    df['price_per_sqft'] = df['price'] * 1000000 / df['total_sqft']

# Remove the price column if it exists (target variable, not a feature)
if 'price' in df.columns:
    df = df.drop('price', axis=1)

# Keep only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

# Encode categorical columns that might be in df
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    numeric_df[col] = le.fit_transform(df[col])

# Make predictions
try:
    predictions = model.predict(numeric_df)
    print("Predictions (first 10):")
    print(predictions[:10])
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Min price: {predictions.min():.2f} Cr")
    print(f"Max price: {predictions.max():.2f} Cr")
    print(f"Average price: {predictions.mean():.2f} Cr")
except Exception as e:
    print(f"Error: {e}")
    print(f"DataFrame columns: {numeric_df.columns.tolist()}")
