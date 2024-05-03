import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler  # Add this import statement

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

# Define base models
base_models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    SVC(),
    LogisticRegression(max_iter=1000),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB()
]

# Train and evaluate base models
for model in base_models:
    if model.__class__.__name__ == 'LogisticRegression':
        # Scale the data for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Logistic Regression with scaled data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Train other models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='fraud', zero_division=0)  # Add zero_division parameter
    recall = recall_score(y_test, y_pred, pos_label='fraud', zero_division=0)  # Add zero_division parameter
    f1 = f1_score(y_test, y_pred, pos_label='fraud', zero_division=0)  # Add zero_division parameter
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print("\n")
