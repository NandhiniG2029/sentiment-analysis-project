import joblib
import numpy as np

# Load Model & Vectorizer
model = joblib.load(r"C:\final year project\back_end\sentiment_model.pkl")
vectorizer = joblib.load(r"C:\final year project\back_end\vectorizer.pkl")

# Function to Predict Sentiment with Probability Adjustment
def predict_sentiment(comment):
    processed_comment = vectorizer.transform([comment])  
    prediction_probs = model.predict_proba(processed_comment)[0]  # Get probabilities

    # Get highest probability class
    predicted_label = np.argmax(prediction_probs) - 1  # Adjusting index for (-1, 0, 1)

    # Neutral Handling: If confidence is low (e.g., < 50%), classify as neutral
    if max(prediction_probs) < 0.5:
        predicted_label = 0  

    sentiment_labels = {0: "Negative (0)", 1: "Neutral (1)", 2: "Positive (2)"}
    return sentiment_labels.get(predicted_label, "Unknown")

# Test Comments
test_comments = [
    "இது ஒரு மிகச் சிறந்த படம்!",  # Positive (1)
    "மிக மோசமான அனுபவம்.",  # Negative (-1)
    "சாதாரணமான படம்.",  # Neutral (0)
    "நல்ல திரைக்கதை மற்றும் இசை.",  # Positive (1)
    "பேசிக்கொள்வதற்கே நல்லது.",  # Neutral (0)
]

# Predict Sentiment for Each Comment
for comment in test_comments:
    print(f"Comment: {comment} → Sentiment: {predict_sentiment(comment)}")