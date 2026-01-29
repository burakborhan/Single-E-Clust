import os
import re
import html
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# If you are running this for the first time and have an internet connection, uncomment these lines:
# nltk.download('stopwords')
# nltk.download('wordnet')

# Stop-word list and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = html.unescape(text)                         # HTML entity decoding
    text = re.sub(r'<[^>]+>', '', text)                 # remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', '', text)  # remove emails
    text = re.sub(r'[@#]\w+', '', text)                 # remove @mentions and #hashtags
    text = re.sub(r'[:;=8][\-^]?[()DPp/\\]', '', text)  # remove emoticons
    text = text.replace('-', ' ')                       # convert hyphens to spaces
    text = re.sub(r'([^\w\s])\1+', '', text)            # remove repeated punctuation
    text = re.sub(r'\(\s*[^\w\s]+\s*\)', '', text)      # remove parentheses containing only punctuation
    text = text.replace('(', '').replace(')', '')       # remove any remaining parentheses
    text = re.sub(r'\s+', ' ', text).strip()            # normalize whitespace

    tokens = []
    for word in text.split():
        if word.isascii() and word not in stop_words and len(word) > 1:
            tokens.append(lemmatizer.lemmatize(word))
    return ' '.join(tokens)

def ensure_period(text: str) -> str:
    return text if text.endswith('.') else text + '.'

datasets = [
    {
        "path": r"C:\Users\Burak\Desktop\final_datasets\bbc\bbc-text.csv",
        "text_cols": ["text"],
        "label_col": "category"
    },
    {
        "path": r"C:\Users\Burak\Desktop\final_datasets\ag_news_dataset\ag_news_4000\train_4000.csv",
        "text_cols": ["Title", "Description"],
        "label_col": "Class Index"
    },
    {
        "path": r"C:\Users\Burak\Desktop\final_datasets\huffpost news category\small_data\news_balanced_500.csv",
        "text_cols": ["headline", "short_description"],
        "label_col": "category"
    },
    {
        "path": r"C:\Users\Burak\Desktop\ek veriseti\yahoo\yahoo_test_1000_per_score.csv",
        "text_cols": ["Question Title", "Question Content", "Best Answer"],
        "label_col": "Score"
    },
]

# Çıktı klasörü
output_dir = r"C:\Users\Burak\Desktop\temiz_veri"
os.makedirs(output_dir, exist_ok=True)


for ds in datasets:
    df = pd.read_csv(ds["path"])

    # Metin sütunlarını birleştir
    combined_text = (
        df[ds["text_cols"]]
        .astype(str)
        .agg(" ".join, axis=1)
    )

    # Temizle
    clean_text = combined_text.apply(preprocess_text).apply(ensure_period)

    # Standart çıktı şeması
    df_clean = pd.DataFrame({
        "text": clean_text,
        "label": df[ds["label_col"]]
    })

    # Dosya adı
    base_name = os.path.splitext(os.path.basename(ds["path"]))[0]
    output_path = os.path.join(output_dir, f"{base_name}_clean.csv")

    df_clean.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Kaydedildi: {output_path}")