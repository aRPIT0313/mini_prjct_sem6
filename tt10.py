import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv("merged_dataset.csv")

# Convert mode_of_transaction values to lowercase
data['mode_of_transaction'] = data['mode_of_transaction'].str.lower()

# Identify top 25 most frequent merchants and places for both legitimate and fraudulent transactions
top_legit_merchants = data[data['label'] == 'legit']['merchant'].value_counts().head(25).index.tolist()
top_legit_places = data[data['label'] == 'legit']['place_of_transaction'].value_counts().head(25).index.tolist()
top_fraud_merchants = data[data['label'] == 'fraud']['merchant'].value_counts().head(25).index.tolist()
top_fraud_places = data[data['label'] == 'fraud']['place_of_transaction'].value_counts().head(25).index.tolist()

# Feature Engineering: Create features indicating whether the transaction involves one of the top merchants or places
data['top_merchant'] = data['merchant'].apply(lambda x: 1 if x in top_legit_merchants + top_fraud_merchants else 0)
data['top_place_of_transaction'] = data['place_of_transaction'].apply(lambda x: 1 if x in top_legit_places + top_fraud_places else 0)

# Features: Include amount_transferred, amount_before_transaction, top_merchant, top_place_of_transaction, and mode_of_transaction
X = data[['amount_transferred', 'amount_before_transaction', 'top_merchant', 'top_place_of_transaction', 'mode_of_transaction']]
y = data['label']  # Target variable

# One-hot encode mode_of_transaction column
X = pd.get_dummies(X, columns=['mode_of_transaction'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Logistic Regression with scaled data
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_scaled, y_train)

# Save the trained model to a file
joblib.dump(logistic_regression_model, 'model.pkl')

# Print confirmation message
print("Logistic Regression model trained and saved as model.pkl")
