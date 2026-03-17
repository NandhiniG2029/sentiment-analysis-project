import pandas as pd  
import joblib  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  

# Load processed test data  
file_path = r"C:\final year project\back_end\processed_tamil_comments.csv"
df = pd.read_csv(file_path, encoding="utf-8")  

# Load saved model & vectorizer  
model = joblib.load(r"C:\final year project\back_end\sentiment_model.pkl")  
vectorizer = joblib.load(r"C:\final year project\back_end\vectorizer.pkl")  

# Transform test data  
X_test = vectorizer.transform(df['processed_text'])  
y_test = df['Sentiment']  

# Predict  
y_pred = model.predict(X_test)  

# **Evaluate Accuracy**  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Model Accuracy: {accuracy:.2f}")  

# **Classification Report**  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  

# **Confusion Matrix**  
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))