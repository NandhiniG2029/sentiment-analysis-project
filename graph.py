import pandas as pd 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load Dataset & Model
file_path = r"C:\final year project\back_end\processed_tamil_comments.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Ensure Sentiment Labels are Integers
df['Sentiment'] = df['Sentiment'].astype(int)

# Load Best Model & Vectorizer
model = joblib.load(r"C:\final year project\back_end\sentiment_model.pkl")
vectorizer = joblib.load(r"C:\final year project\back_end\vectorizer.pkl")

# Transform Comments Using Vectorizer
X = vectorizer.transform(df['processed_text'])
y_true = df['Sentiment']

# Predict Sentiments
y_pred = model.predict(X)

# Sentiment Mapping
sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}



# 🎯 Plot Sentiment Distribution
plt.figure(figsize=(8,5))
sns.countplot(x=df['Sentiment'].map(sentiment_labels),
              order=["Negative", "Neutral", "Positive"],
              palette=["red", "gray", "green"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 🎯 Confusion Matrix
from sklearn.utils.multiclass import unique_labels

# Define label order
label_order = [-1, 0, 1]
cm = confusion_matrix(y_true, y_pred, labels=label_order)

plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[sentiment_labels[i] for i in label_order],
            yticklabels=[sentiment_labels[i] for i in label_order])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


