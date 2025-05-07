# 📰 NewsInsight Pro

**NewsInsight Pro** is a powerful Streamlit-based web application that delivers a **personalized news discovery experience** using real-time data from [NewsAPI.org](https://newsapi.org). It supports **semantic similarity recommendations**, **AI-driven analytics**, and interactive visualizations to help users explore, search, and engage with news more meaningfully.

---

## 🚀 Features

* **Real-Time News Feed**: Pulls live articles using the NewsAPI based on category, country, and custom search queries.
* **Smart Recommendations**: Uses semantic embeddings (`SentenceTransformer`) and cosine similarity to recommend similar articles you've liked.
* **Advanced Filters**: Filter news by reading time, country, source exclusion, sentiment, and timeframe.
* **AI-Driven Analytics**:

  * Article sentiment distribution
  * Keyword/topic trending chart
  * Top sources pie chart
  * Publication timeline
* **Custom UI/UX**: Styled with CSS for a modern, card-based experience with hover effects, badges, and clear readability.
* **User API Key Support**: Switch between default API and your own NewsAPI key for unrestricted access.
* **Session-Based Personalization**: Tracks likes, preferences, and view history across sessions.

---

## 📂 Project Structure

```
.
├── app2.py                # Main Streamlit application
├── README.md              # You're reading it
└── requirements.txt       # Python dependencies
```

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit + Plotly + Custom CSS
* **Backend**: Python
* **NLP**: `sentence-transformers (MiniLM-L6-v2)` for semantic understanding
* **Data Source**: [NewsAPI.org](https://newsapi.org)
* **Libraries**:

  * `requests`, `pandas`, `numpy`, `sklearn`, `plotly`, `PIL`, `io`, `datetime`, `random`

---

## 📦 Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/Akanksharao24/newsinsight-pro.git
cd newsinsight-pro
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run app2.py
```

4. **(Optional) Get a Free NewsAPI Key**

Register at [https://newsapi.org/register](https://newsapi.org/register) and paste your API key in the sidebar.

---

## 🧠 How It Works

### 🔍 Fetching News

The app calls either:

* `https://newsapi.org/v2/top-headlines` (for category-based headlines), or
* `https://newsapi.org/v2/everything` (for custom queries)

Articles are cached hourly and include:

* Title
* Description
* URL & image
* Source name
* Reading time (estimated by word count)
* Sentiment (simulated for demo purposes)
* Popularity (randomized for demo use)

### ❤️ Recommendation Engine

When you like an article:

* It’s stored in `session_state.liked_articles`
* Its `title + description` is embedded using `all-MiniLM-L6-v2`
* Cosine similarity scores are calculated against other articles
* Top N similar articles are recommended with a similarity badge

### 📊 Insights Tab

Contains:

* Top keywords (based on TF)
* Sentiment histogram
* Timeline of publication dates
* Source distribution (pie chart)
* Word cloud for popular topics

---

## 📤 API Key Usage

* By default, the app uses a **demo key** (`DEFAULT_API_KEY`).
* To avoid rate limits and access full international coverage:

  1. Get your own key from [NewsAPI](https://newsapi.org/register)
  2. Paste it in the **API Key** field in the sidebar

---

## 🔐 API Endpoints Used

| Endpoint            | Description                             |
| ------------------- | --------------------------------------- |
| `/v2/top-headlines` | For localized, category-based headlines |
| `/v2/everything`    | For full-text search, with filters      |

> Rate limits and country restrictions apply for the free tier. Use your personal key for unrestricted access.

---

## 🧪 Testing

Test your NewsAPI key via:

```python
test_api_key(api_key)
```

It validates against `/v2/everything?q=test`.

---

## 🧰 Requirements

```txt
streamlit
requests
pandas
numpy
plotly
sentence-transformers
scikit-learn
Pillow
```

---

## 🧠 Future Ideas

* Add login support for persistent preferences
* Integrate real-time sentiment analysis using HuggingFace models
* Add bookmarking and user profiles
* Serverless or Docker deployment
* Email digests or Telegram bot support

---

## 🎬 Demo Video

[![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE)

> Replace `YOUR_VIDEO_ID_HERE` with your uploaded demo link (YouTube, Loom, Drive).

---

## 🧑‍💻 Author

**Developed by:** \[Your Name]
**LinkedIn:** \[Your LinkedIn]
**GitHub:** \[Your GitHub]
**Email:** \[[your.email@example.com](mailto:your.email@example.com)]

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


---
