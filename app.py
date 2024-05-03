from flask import Flask, request, render_template
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("merged_dataset.csv")

# Load the trained Logistic Regression model
logistic_regression_model = joblib.load('model.pkl')

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

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = {
            'amount_transferred': float(request.form['amount_transferred']),
            'mode_of_transaction': request.form['mode_of_transaction'],
            'date_and_time': request.form['date_and_time'],
            'account_number': float(request.form['account_number']),
            'place_of_transaction': request.form['place_of_transaction'],
            'merchant': request.form['merchant'],
            'amount_before_transaction': float(request.form['amount_before_transaction']),
            'amount_after_transaction': float(request.form['amount_after_transaction'])
        }

        # Convert input merchant and dataset values to lowercase for case-insensitive comparison
        input_merchant_lower = input_data['merchant'].lower()
        top_fraud_merchants_lower = [merchant.lower() for merchant in top_fraud_merchants]
        top_legit_merchants_lower = [merchant.lower() for merchant in top_legit_merchants]

        # Convert input place of transaction and dataset values to lowercase for case-insensitive comparison
        input_place_lower = input_data['place_of_transaction'].lower()
        top_fraud_places_lower = [place.lower() for place in top_fraud_places]
        top_legit_places_lower = [place.lower() for place in top_legit_places]

        # Check if input merchant is in top fraudulent merchants
        if input_merchant_lower in top_fraud_merchants_lower:
            return render_template('result.html', prediction="fraud")

        # Check if input merchant is in top legitimate merchants
        elif input_merchant_lower in top_legit_merchants_lower:
            return render_template('result.html', prediction="legitimate")

        # Check if input place of transaction is in top fraudulent places
        elif input_place_lower in top_fraud_places_lower:
            return render_template('result.html', prediction="fraud")

        # Check if input place of transaction is in top legitimate places
        elif input_place_lower in top_legit_places_lower:
            return render_template('result.html', prediction="legitimate")

        else:
            # Fuzzy matching with dataset
            max_similarity = 0
            matching_row = None

            for _, row in data.iterrows():
                similarity = fuzz.partial_ratio(str(row['amount_transferred']), str(input_data['amount_transferred']))
                similarity += fuzz.partial_ratio(row['mode_of_transaction'], input_data['mode_of_transaction'])
                similarity += fuzz.partial_ratio(str(row['account_number']), str(input_data['account_number']))
                similarity += fuzz.partial_ratio(row['place_of_transaction'], input_data['place_of_transaction'])
                similarity += fuzz.partial_ratio(row['merchant'], input_data['merchant'])
                similarity += fuzz.partial_ratio(str(row['amount_before_transaction']), str(input_data['amount_before_transaction']))
                similarity += fuzz.partial_ratio(str(row['amount_after_transaction']), str(input_data['amount_after_transaction']))

                if similarity > max_similarity:
                    max_similarity = similarity
                    matching_row = row

            if matching_row is not None:
                prediction_label = matching_row['label']
                return render_template('result.html', prediction=prediction_label)
            
            else:
                # Preprocess input data for model prediction
                input_df = pd.DataFrame([input_data])
                input_df['mode_of_transaction'] = input_df['mode_of_transaction'].str.lower()
                input_df_scaled = StandardScaler().fit_transform(input_df)

                # Predict using the model
                prediction = logistic_regression_model.predict(input_df_scaled)[0]

                return render_template('result.html', prediction=prediction)

    except Exception as e:
        error_message = "An error occurred: {}".format(str(e))
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
