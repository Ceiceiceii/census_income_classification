import pickle
import pandas as pd
import numpy as np


NUMERIC_FEATURES = {
    'age',
    'capital gains',
    'capital losses',
    'weeks worked in year',
    'num persons worked for employer',
}

def load_trained_model():
    with open('simple_downsampled_model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

def create_sample_customers():
    sample_data = {
        'age': [39, 20, 45, 60],
        'class of worker': ['Not in universe', 'Not in universe', 'Self-employed-not incorporated', 'Private'],
        'education': [
            'Bachelors degree(BA AB BS)', 
            'Bachelors degree(BA AB BS)', 
            'Masters degree(MA MS MEng MEd MSW MBA)', 
            'Some college but no degree'
        ],
        'marital stat': [
            'Married-civilian spouse present', 
            'Never married', 
            'Married-civilian spouse present', 
            'Divorced'
        ],
        'major occupation code': [
            'Not in universe', 
            'Handlers equip cleaners etc', 
            'Professional specialty', 
            'Adm support including clerical'
        ],
        'race': ['White', 'Black', 'Asian or Pacific Islander', 'White'],
        'sex': ['Male', 'Female', 'Male', 'Female'],
        'capital gains': [0, 1000, 5178, 0],
        'capital losses': [1000, 0, 800, 300],
        'weeks worked in year': [52, 30, 50, 48],
        'num persons worked for employer': [1, 0, 3, 1],
    }
    return pd.DataFrame(sample_data)


def predict_customer_income(model_dict, customer_df):
    
    model = model_dict['model']
    feature_names = model_dict['selected_features'] 

    print("predicting income...")
    predictions = model.predict(customer_df)
    probabilities = model.predict_proba(customer_df)[:, 1]

    return predictions, probabilities

def main():
    print("="*60)
    print("WALMART INCOME CLASSIFIER DEMONSTRATION")
    print("="*60)

    # load model package
    model_dict = load_trained_model()
    print("Model loaded")
    print(f" Model type: {model_dict['model_name']}")
    print(f" Scaler mode: {model_dict.get('scaler_mode','(n/a)')}")
    print(f" Features used: {len(model_dict['selected_features'])}")
    perf = model_dict.get('performance', {})
    if perf:
        print(f" Test Accuracy: {perf.get('accuracy', float('nan')):.4f}")
        print(f" Test ROC-AUC: {perf.get('roc_auc', float('nan')):.4f}")
        print(f" Test PR-AUC : {perf.get('pr_auc', float('nan')):.4f}")

    customers = create_sample_customers()
    print(f"\nCreated {len(customers)} sample customer profiles")

    # Predict
    preds, probs = predict_customer_income(model_dict, customers)

    # Display results
    print("\nCUSTOMER INCOME PREDICTIONS")
    for i, (pred, p) in enumerate(zip(preds, probs), start=1):
        income_pred = ">$50K" if pred == 1 else "≤$50K"
        confidence = p * 100
        row = customers.iloc[i-1]
        print(f"Customer {i}:")
        print(f"  Age: {row['age']}, Education: {row['education'][:28]}...")
        print(f"  Predicted Income: {income_pred} (Confidence: {confidence:.1f}%)\n")

if __name__ == "__main__":
    main()
