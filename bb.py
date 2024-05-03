import pandas as pd
data = pd.read_csv("merged_dataset.csv")

# Get top fraudulent and legitimate merchants
fraud_merchants = data[data['label'] == 'fraud']['merchant'].value_counts().head(20).index.tolist()
legit_merchants = data[data['label'] == 'legitimate']['merchant'].value_counts().head(20).index.tolist()

# Get top fraudulent and legitimate places of transaction
fraud_places = data[data['label'] == 'fraud']['place_of_transaction'].value_counts().head(20).index.tolist()
legit_places = data[data['label'] == 'legitimate']['place_of_transaction'].value_counts().head(20).index.tolist()

# Get unique fraudulent and legitimate merchants and places of transaction
top_fraud_merchants = list(set(fraud_merchants) - set(legit_merchants))
top_legit_merchants = list(set(legit_merchants) - set(fraud_merchants))
top_fraud_places = list(set(fraud_places) - set(legit_places))
top_legit_places = list(set(legit_places) - set(fraud_places))

# Print the unique values
print("Unique fraudulent merchants:\n", top_fraud_merchants)
print("Unique legitimate merchants:\n", top_legit_merchants)
print("Unique fraudulent places of transaction:\n", top_fraud_places)
print("Unique legitimate places of transaction:\n", top_legit_places)
