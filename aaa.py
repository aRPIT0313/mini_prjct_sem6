import pandas as pd
data = pd.read_csv("merged_dataset.csv")

# Define top fraudulent and legitimate merchants
top_fraud_merchants = data[data['label'] == 'fraud']['merchant'].value_counts().head(20).index.tolist()
top_legit_merchants = data[data['label'] == 'legitimate']['merchant'].value_counts().head(20).index.tolist()

# Define top fraudulent and legitimate places of transaction
top_fraud_places = data[data['label'] == 'fraud']['place_of_transaction'].value_counts().head(20).index.tolist()
top_legit_places = data[data['label'] == 'legitimate']['place_of_transaction'].value_counts().head(20).index.tolist()

# Print the most frequent legitimate and fraudulent merchants and places of transaction
print("Top 20 Most Frequent Legitimate Merchants:")
print(top_legit_merchants)
print("\nTop 20 Most Frequent Fraudulent Merchants:")
print(top_fraud_merchants)
print("\nTop 20 Most Frequent Legitimate Places of Transaction:")
print(top_legit_places)
print("\nTop 20 Most Frequent Fraudulent Places of Transaction:")
print(top_fraud_places)
