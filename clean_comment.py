import pandas as pd
import re

# Load the collected data
excel_path = r"C:\final year project85\back_end\youtube_comments.xlsx"
df = pd.read_excel(excel_path)

# Function to check if a comment is in Tamil
def is_tamil(text):
    return bool(re.search(r'[\u0B80-\u0BFF]', text))  # Tamil Unicode range

# Function to clean the comments
def clean_comment(comment):
    comment = re.sub(r'[^ \u0B80-\u0BFF]', '', comment)  # Keep only Tamil characters and spaces
    comment = re.sub(r'\s+', ' ', comment).strip()  # Remove extra spaces
    return comment

# Filter only Tamil comments and clean them
df["Comment"] = df["Comment"].astype(str)  # Ensure all comments are strings
df = df[df["Comment"].apply(is_tamil)]  # Keep only Tamil comments
df["Comment"] = df["Comment"].apply(clean_comment)  # Clean special characters and emojis

# Save the cleaned Tamil comments
cleaned_excel_path = r"C:\final year project85\back_end\cleaned_tamil_comments.xlsx"
df.to_excel(cleaned_excel_path, index=False)

print(f"Total Tamil comments saved: {len(df)}")
print(f"Cleaned Tamil comments saved to: {cleaned_excel_path}")
