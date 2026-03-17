import subprocess
import json
import pandas as pd
import re  # Importing regex module

# Function to clean comments (Optional)
def clean_comment(comment):
    comment = re.sub(r'\s+', ' ', comment)  # Replace multiple spaces/newlines with a single space
    comment = comment.strip()
    return comment

# List of YouTube video URLs
video_urls = [
    "https://youtu.be/cAz5o6_h2Wg?si=EYUqeD1C-PWrPLKy",
    "https://youtu.be/vX5wjPHStjU?si=E7yLz8tbvyT1jKfr",
    "https://youtu.be/ftpUv5UBmic?si=A0_RyzumPBUbNCbv",
    "https://youtu.be/DsL_127Tc0Q?si=NOQXn4TWRIzQwvJu",
    "https://youtu.be/LtN5ZE2Y_gg?si=cGc5lPt7DlPcKHMH",
    "https://youtu.be/6YWbb3Ozb7k?si=Y1XA3unKL_X2SfLi",
    "https://youtu.be/WFw1jIrl2lg?si=qOMt0J6VQ0Meh0c9",
    "https://youtu.be/En6TdrphpEM?si=l0GAMiwz_uxgIGxS",
    "https://youtu.be/FnS38Ko9YQA?si=OvrKiZM87c66RR30",
    "https://youtu.be/gvzFPJx3Cuk?si=jZQ5mN4bVvzMdGYm",
    "https://youtu.be/Se6luqD2mS8?si=sDwnHrojEGTGi8fc",
    "https://youtu.be/8IhZynj_Fy0?si=eCdHA1ZONTUx7rAt",
    "https://youtu.be/7PhlzLZcpMk?si=EhWToq4TzYOU4GrV"
]

# List to store comments
all_comments = []

for video_url in video_urls:
    print(f"Processing video: {video_url}")
    command = f'yt-dlp --skip-download --write-comments --extractor-args youtube:max_comments=100 -j "{video_url}"'
    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    try:
        data = json.loads(output.stdout)
        comments = data.get("comments", [])
        for comment in comments:
            cleaned_comment = clean_comment(comment["text"])
            all_comments.append([cleaned_comment])  # Only one column

    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON data for video: {video_url}")

# Convert to DataFrame
df = pd.DataFrame(all_comments, columns=["Comment"])  # Only user comments

# Save to Excel file
excel_path = r"C:\final year project85\back_end\youtube_comments.xlsx"
df.to_excel(excel_path, index=False)

print(f"Total comments collected: {len(all_comments)}")
print(f"Comments saved to: {excel_path}")
