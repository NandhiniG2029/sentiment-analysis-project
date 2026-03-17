import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set the correct file path
file_path = r"C:\final year project85\back_end\labeled_tamil_comments.csv"

# Load the CSV file
df = pd.read_csv(file_path, encoding="utf-8")

# Display first few rows
print(df.head())

# Tokenization using simple split (instead of indicnlp.tokenize)
df['tokens'] = df['Comment'].apply(lambda x: str(x).strip().split())

# Display tokenized data
print(df[['Comment', 'tokens']].head())

# Show original vs tokenized
for i in range(5):  
    print(f"Original: {df['Comment'][i]}")
    print(f"Tokenized: {df['tokens'][i]}")
    print("-" * 50)

# Define Tamil stopwords
tamil_stopwords = set(["இது", "ஒரு", "என்று", "நான்", "என்", "உள்ளது", "இல்லை"])

# Remove stopwords
df['filtered_tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in tamil_stopwords])

# Display tokens after stopword removal
print(df[['tokens', 'filtered_tokens']].head())

# Convert tokens to a single processed text column
df['processed_text'] = df['filtered_tokens'].apply(lambda x: ' '.join(x))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Print feature matrix shape
print("Feature matrix shape:", X.shape)

# Save the processed data
output_file = r"C:\final year project85\back_end\processed_tamil_comments.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("✅ Processed data saved successfully!")
