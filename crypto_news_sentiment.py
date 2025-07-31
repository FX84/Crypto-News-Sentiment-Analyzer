# crypto_news_sentiment.py

import requests
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

# Инициализация анализатора тональности
nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()

# ==== CONFIG ====
API_KEY = "YOUR_API_KEY_HERE"
LIMIT = 30  # Количество новостей
SHOW_TOP = 10  # Сколько заголовков выводить
SAVE_CSV = True  # Сохранить в CSV
# ===============

def get_news(limit=30):
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса: {response.status_code}")
    data = response.json()
    return data["results"][:limit]

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return compound, label

def main():
    print("🔄 Получение новостей...")
    news_items = get_news(LIMIT)

    data = []
    for item in news_items:
        title = item["title"]
        url = item["url"]
        published = item["published_at"]
        score, label = analyze_sentiment(title)
        data.append({
            "title": title,
            "url": url,
            "published": published,
            "sentiment_score": score,
            "sentiment_label": label
        })

    df = pd.DataFrame(data)
    
    # Подсчёт
    counts = df["sentiment_label"].value_counts()
    print(f"\n🔹 Total News Articles: {len(df)}")
    print(f"✅ Positive: {counts.get('Positive', 0)}")
    print(f"⚪ Neutral: {counts.get('Neutral', 0)}")
    print(f"❌ Negative: {counts.get('Negative', 0)}\n")

    print(f"📰 Top {SHOW_TOP} headlines:")
    for i, row in df.head(SHOW_TOP).iterrows():
        symbol = {"Positive": "[+]", "Neutral": "[0]", "Negative": "[-]"}.get(row["sentiment_label"], "[?]")
        print(f"{symbol} {row['title']}")

    # График
    plt.figure(figsize=(6, 4))
    df["sentiment_label"].value_counts().plot(kind="bar")
    plt.title("Распределение новостей по тональности")
    plt.xlabel("Категория")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.show()

    # Сохранение CSV
    if SAVE_CSV:
        filename = f"crypto_news_sentiment_{datetime.date.today()}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Сохранено в файл: {filename}")

if __name__ == "__main__":
    main()