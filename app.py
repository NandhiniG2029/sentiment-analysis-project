from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os

app = Flask(__name__)

# Load Model and Vectorizer with Error Handling
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("sentiment_model.pkl")
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit(1)

# Store uploaded data globally
uploaded_data = []
uploaded_df = None  # Store dataset for report generation

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    global uploaded_data, uploaded_df
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            try:
                df = pd.read_csv(file)
                if not {"Comment", "Sentiment"}.issubset(df.columns):
                    return jsonify({"error": "Invalid file format. Ensure 'Comment' and 'Sentiment' columns exist."}), 400
                
                uploaded_data = df["Comment"].dropna().tolist()
                uploaded_df = df  # Store for report generation
                return redirect(url_for("select"))
            except Exception as e:
                return jsonify({"error": f"Error reading file: {str(e)}"}), 500
    return render_template("upload.html")

@app.route("/select", methods=["GET", "POST"])
def select():
    global uploaded_data
    if not uploaded_data:
        return redirect(url_for("upload"))
    if request.method == "POST":
        return redirect(url_for("predict"))
    return render_template("select.html", comments=uploaded_data)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()
            comments = data.get("comments", [])

            if not comments:
                return jsonify({"error": "No comments provided"}), 400

            comment_vectorized = vectorizer.transform(comments)
            predictions = model.predict(comment_vectorized)

            sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

            results = []
            for comment, pred in zip(comments, predictions):
                sentiment_text = sentiment_labels.get(pred, "Unknown")
                results.append({
                    "comment": comment,
                    "sentiment": sentiment_text
                })

            return jsonify({"predictions": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("predict.html")


@app.route("/report", methods=["GET", "POST"])
def report():
    global uploaded_df
    if uploaded_df is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    # Extract and clean data
    df = uploaded_df.copy()
    df.dropna(subset=["Comment", "Sentiment"], inplace=True)

    # Convert Sentiment to numeric safely
    df["Sentiment"] = pd.to_numeric(df["Sentiment"], errors="coerce").dropna().astype(int)

    # Ensure there are valid sentiment labels
    if df.empty:
        return jsonify({"error": "No valid sentiment data available"}), 400

    # Vectorizing comments
    X = df["Comment"]
    X_vectorized = vectorizer.transform(X)

    # Predicting sentiments
    y_pred = model.predict(X_vectorized).astype(int)

    # Generate classification report
    report = classification_report(df["Sentiment"], y_pred, output_dict=True)

    # Generate report text
    report_text = "📊 Sentiment Analysis Report\n\n"
    report_text += f"Total Comments: {len(df)}\n"
    report_text += f"Sentiment Distribution:\n{df['Sentiment'].value_counts().to_string()}\n\n"
    # report_text += "🔹 Model Performance\n"
    # report_text += f"Accuracy: {report['accuracy']:.2f}\n"
    # report_text += f"Precision: {report['weighted avg']['precision']:.2f}\n"
    # report_text += f"Recall: {report['weighted avg']['recall']:.2f}\n"
    # report_text += f"F1-score: {report['weighted avg']['f1-score']:.2f}\n\n"

    # Save report with UTF-8 encoding to avoid Unicode issues
    report_file = "sentiment_report.txt"
    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_text)

    return send_file(report_file, as_attachment=True)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        feedback_text = request.form.get("feedback")
        if feedback_text:
            with open("feedback.txt", "a", encoding="utf-8") as f:
                f.write(feedback_text + "\n")
            return redirect(url_for("home"))
    return render_template("feedback.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
