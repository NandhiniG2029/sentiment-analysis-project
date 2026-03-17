import pandas as pd  
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  

# Load Processed Dataset (After Feature Extraction)
file_path = r"C:\final year project85\back_end\tamil_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8")  

# Ensure Sentiment Labels are Integers (-1, 0, 1)
df['Sentiment'] = df['Sentiment'].astype(int)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words=None, ngram_range=(1,2))
X = vectorizer.fit_transform(df['Comment'])  
y = df['Sentiment']  

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(class_weight="balanced")  # Handles Imbalance
}

# Train & Evaluate Models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model  # Store Best Model

# Save Best Model & Vectorizer
joblib.dump(best_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"\n🎯 Best Model Saved: {best_model.__class__.__name__} with Accuracy: {best_accuracy:.2f}")
print("✅ Model & Vectorizer Saved Successfully!")

# Display Sentiment Distribution
print("\nSentiment Distribution:")
print(f"Positive (2)  : {df['Sentiment'].value_counts().get(2, 0)}")
print(f"Negative (0) : {df['Sentiment'].value_counts().get(0, 0)}")
print(f"Neutral (1)   : {df['Sentiment'].value_counts().get(1, 0)}")
