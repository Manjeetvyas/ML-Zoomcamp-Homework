import pickle

# Step 2: Open and load the saved pipeline model
with open('pipeline.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Step 3: Put the lead data into a Python dictionary
lead = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

# Step 4 & 5: Call predict_proba with a list of dicts and extract class 1 probability
y_pred = pipeline.predict_proba([lead])
probability = y_pred[0][1]

# Step 6: Round to 3 decimal places
result = round(probability, 3)
print(f"Conversion probability: {result}")