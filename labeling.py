import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Load the CSV file
file_path = "tamil_dataset.csv"
df = pd.read_csv(file_path)

# Ensure correct column name
comment_column = "Comment"  # Use exact column name from your file

# Create a new column for sentiment labels
df["Sentiment"] = None

# Process each comment
for index, row in df.iterrows():
    comment = row[comment_column]
    if pd.notna(comment):  # Check if the comment is not empty
        try:
            result = sentiment_pipeline(comment)[0]['label']

            # Convert result to numeric sentiment values
            if "5 stars" in result or "4 stars" in result:
                sentiment = 2# Positive
            elif "3 stars" in result:
                sentiment = 1 # Neutral
            else:
                sentiment = 0  # Negative

            df.at[index, "Sentiment"] = sentiment  # Store sentiment value
        except Exception as e:
            df.at[index, "Sentiment"] = "Error"

# Save the labeled file
output_file = "tamil_dataset.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Sentiment labeling completed! File saved as '{output_file}'")