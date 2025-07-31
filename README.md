# 📊 Crypto News Sentiment Analyzer

Анализатор тональности криптовалютных новостей на Python с использованием VADER и API CryptoPanic.

---

## 🔍 Описание

Скрипт собирает свежие крипто-новости через [CryptoPanic API](https://cryptopanic.com/developers/api/), анализирует эмоциональную окраску (тональность) заголовков и визуализирует распределение по категориям: **позитивные**, **нейтральные**, **негативные**.

Это полезный инструмент для крипто-инвесторов, аналитиков и разработчиков, желающих быстро понять общее настроение на рынке.

---

## 📦 Возможности

- 📥 Получение новостей через CryptoPanic API
- 🧠 Анализ заголовков с использованием `NLTK` и `VADER`
- 📊 График распределения новостей по тональности
- 📰 Консольный вывод заголовков с пометками [+]/[0]/[-]
- 💾 Сохранение результатов в `.csv` файл

---

## 📸 Пример вывода

```bash
🔹 Total News Articles: 30
✅ Positive: 12
⚪ Neutral: 10
❌ Negative: 8

📰 Top 10 headlines:
[+] Bitcoin surges above $60K for the first time since 2021
[0] Ethereum shows sideways movement amid mixed signals
[-] SEC crackdown creates panic among DeFi investors
...
```

И будет отображена вот такая гистограмма:

```
Positive |████████████
Neutral  |█████████
Negative |████████
```

---

## 🚀 Установка и запуск

### 1. Клонируй репозиторий

```bash
git clone https://github.com/your-username/crypto-news-sentiment.git
cd crypto-news-sentiment
```

### 2. Установи зависимости

```bash
pip install -r requirements.txt
```

Или вручную:

```bash
pip install requests pandas matplotlib nltk
```

### 3. Получи API-ключ

- Зарегистрируйся на [CryptoPanic](https://cryptopanic.com)
- Перейди в **Account → API Token**
- Скопируй свой API ключ

### 4. Запусти скрипт

```bash
python crypto_news_sentiment.py
```

🔧 Не забудь заменить строку в скрипте:

```python
API_KEY = "YOUR_API_KEY_HERE"
```

---

## 📁 Структура проекта

```bash
crypto-news-sentiment/
├── crypto_news_sentiment.py   # Главный скрипт
├── README.md                  # Документация
└── requirements.txt           # Зависимости
```

---

## 🛠 Возможные улучшения

- Добавление поддержки ключевых слов (например, только BTC или ETH)
- Поддержка других API (например, CoinDesk, NewsAPI)
- Web-интерфейс (с Flask или Streamlit)
- Telegram-бот, присылающий ежедневный отчёт

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
