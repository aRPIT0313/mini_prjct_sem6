import pandas as pd

# Read the datasets
bank_df = pd.read_csv("bank_transaction_dataset.csv")
credit_card_df = pd.read_csv("credit_card_transaction_dataset.csv")

# Rename columns of the second dataset to match the first dataset
credit_card_df.rename(columns={
    'transaction_amount': 'amount_transferred',
    'transaction_date_time': 'date_and_time',
    'card_number': 'account_number',
}, inplace=True)

# Add mode_of_transaction column to the second dataset
credit_card_df['mode_of_transaction'] = 'card'

# Add place_of_transaction column to the second dataset
credit_card_df['place_of_transaction'] = 'unknown'

# Add merchant column to the first dataset
bank_df['merchant'] = 'unknown'

# Reorder columns
merged_df = pd.concat([bank_df[['amount_transferred', 'mode_of_transaction', 'date_and_time', 'account_number', 'place_of_transaction', 'merchant', 'label', 'amount_before_transaction', 'amount_after_transaction']],
                       credit_card_df[['amount_transferred', 'mode_of_transaction', 'date_and_time', 'account_number', 'place_of_transaction', 'merchant', 'label', 'amount_before_transaction', 'amount_after_transaction']]], ignore_index=True)

# Save the merged dataset
merged_df.to_csv("merged_dataset.csv", index=False)
