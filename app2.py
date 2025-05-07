import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import time
import random
from PIL import Image
import io

# Minor tweak to test co-author commit
# --- SETTINGS ---
DEFAULT_API_KEY = "ADD_YOUR_API_KEY"
API_KEY = "ADD_YOUR_API_KEY"
MAIN_CATEGORIES = ["business", "entertainment", "general", "health", "science", "sports", "technology"]
SUB_CATEGORIES = {
    "business": ["economy", "startups", "finance", "markets", "entrepreneurship"],
    "entertainment": ["movies", "music", "celebrities", "tv shows", "gaming"],
    "general": ["world", "politics", "environment", "education", "society"],
    "health": ["medicine", "fitness", "nutrition", "mental health", "wellness"],
    "science": ["space", "physics", "biology", "climate", "research"],
    "sports": ["football", "basketball", "tennis", "olympics", "motorsports"],
    "technology": ["ai", "blockchain", "gadgets", "software", "cybersecurity"]
}
COUNTRIES = {
    "United States": "us", "United Kingdom": "gb", "Canada": "ca", "Australia": "au", 
    "India": "in", "Germany": "de", "France": "fr", "Japan": "jp", "Brazil": "br",
    "China": "cn", "Russia": "ru", "South Africa": "za", "Mexico": "mx", "Italy": "it"
}
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"
EVERYTHING_URL = "https://newsapi.org/v2/everything"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NewsInsight Pro", 
    page_icon="📰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .category-badge {
        background-color: #E3F2FD;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        color: #1565C0;
        margin-right: 0.5rem;
    }
    .article-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E0E0E0;
        transition: transform 0.3s;
    }
    .article-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .article-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .article-meta {
        font-size: 0.8rem;
        color: #757575;
        margin-bottom: 0.5rem;
    }
    .article-desc {
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .read-more {
        background-color: #1E88E5;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        text-decoration: none;
        font-size: 0.8rem;
    }
    .read-more:hover {
        background-color: #1565C0;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.3rem;
    }
    .trending-badge {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    .stats-card {
        background-color: #F5F5F5;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stats-label {
        font-size: 0.9rem;
        color: #757575;
    }
    .api-key-input {
        margin-top: 1rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news(category, country, query=None, days=7, page_size=50, api_key=None):
    """Fetch news articles based on filters"""
    try:
        # Use provided API key or default
        current_api_key = api_key if api_key else API_KEY
        
        if query:
            # Search everything endpoint with query - works for all countries
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            params = {
                "q": query,
                "language": "en",
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": page_size,
                "apiKey": current_api_key
            }
            response = requests.get(EVERYTHING_URL, params=params)
        else:
            # For international news, use the everything endpoint with country name as query
            # This is a workaround for the free tier limitations
            if country != "us" and current_api_key == DEFAULT_API_KEY:
                country_name = [k for k, v in COUNTRIES.items() if v == country][0]
                params = {
                    "q": f"{country_name} {category}",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "apiKey": current_api_key
                }
                response = requests.get(EVERYTHING_URL, params=params)
            else:
                # Top headlines by category and country
                params = {
                    "category": category,
                    "country": country,
                    "pageSize": page_size,
                    "apiKey": current_api_key
                }
                response = requests.get(NEWSAPI_URL, params=params)
        
        data = response.json()
        if data.get("status") != "ok":
            error_msg = data.get('message', 'Unknown error')
            st.error(f"API Error: {error_msg}")
            if "apiKey" in error_msg:
                st.warning("Your API key may be invalid or expired. Please check your API key.")
            return pd.DataFrame()
                
        articles = data.get("articles", [])
        if not articles:
            return pd.DataFrame()
            
        df = pd.DataFrame(articles)
        
        # Clean and process data
        df = df.dropna(subset=["title", "url"])
        
        # Convert publishedAt to datetime and handle timezone
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
        df["publishedAt"] = df["publishedAt"].dt.tz_localize(None)  # Make timezone naive
        
        # Add derived fields
        df["publishedDate"] = df["publishedAt"].dt.date
        
        # Calculate days ago safely
        now = datetime.now()
        df["daysAgo"] = [(now - dt).days for dt in df["publishedAt"]]
        
        # Add sentiment score (simulated)
        df["sentiment"] = np.random.uniform(-1, 1, len(df))
        
        # Add reading time estimate - safely handle missing content
        df["content"] = df["content"].fillna("")
        df["readingTime"] = df["content"].apply(lambda x: max(1, len(str(x).split()) // 200))
        
        # Add popularity score (simulated)
        df["popularity"] = np.random.randint(10, 1000, len(df))
        
        return df
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_article_image(url):
    """Get article image or placeholder"""
    if pd.isna(url) or not url:
        # Generate colored placeholder
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        img = Image.new('RGB', (600, 400), color=color)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    try:
        response = requests.get(url, timeout=3)
        return response.content
    except:
        # Generate colored placeholder on error
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        img = Image.new('RGB', (600, 400), color=color)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

def get_recommendations(df, liked_articles, n=5):
    """Get advanced recommendations based on liked articles"""
    if not liked_articles or df.empty:
        return pd.DataFrame()
    
    try:
        # Get embeddings for all articles
        all_titles = df['title'].tolist()
        all_descriptions = df['description'].fillna('').tolist()
        
        # Combine title and description for better context
        all_content = [f"{title}. {desc}" for title, desc in zip(all_titles, all_descriptions)]
        content_embeddings = MODEL.encode(all_content)
        
        # Get embeddings for liked articles
        liked_indices = []
        for title in liked_articles:
            indices = df[df['title'] == title].index
            if not indices.empty:
                liked_indices.append(indices[0])
        
        if not liked_indices:
            return pd.DataFrame()
        
        liked_embeddings = content_embeddings[liked_indices]
        
        # Calculate similarity with all articles
        similarity_matrix = cosine_similarity(liked_embeddings, content_embeddings)
        
        # Average similarity across all liked articles
        avg_similarities = np.mean(similarity_matrix, axis=0)
        
        # Get top N recommendations (excluding already liked articles)
        recommended_indices = []
        sorted_indices = np.argsort(-avg_similarities)
        
        for idx in sorted_indices:
            if df.iloc[idx]['title'] not in liked_articles:
                recommended_indices.append(idx)
                if len(recommended_indices) >= n:
                    break
        
        # Add similarity score to recommendations
        recommendations = df.iloc[recommended_indices].copy() if recommended_indices else pd.DataFrame()
        if not recommendations.empty:
            recommendations['similarity'] = avg_similarities[recommended_indices]
            recommendations = recommendations.sort_values('similarity', ascending=False)
        
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame()

def display_article_card(article, idx, col, show_image=True):
    """Display a news article in a card format"""
    try:
        with col:
            with st.container():
                st.markdown(f"<div class='article-card'>", unsafe_allow_html=True)
                
                # Display image if available
                if show_image and 'urlToImage' in article and article['urlToImage']:
                    image = get_article_image(article['urlToImage'])
                    st.image(image, use_container_width=True)  # Fixed deprecated parameter
                
                # Article title and metadata
                st.markdown(f"<div class='article-title'>{article['title']}</div>", unsafe_allow_html=True)
                
                # Source and date
                published_time = article['publishedAt'].strftime('%b %d, %Y')
                source_name = article.get('source', {}).get('name', 'Unknown')
                
                st.markdown(
                    f"<div class='article-meta'>📰 {source_name} • 🕒 {published_time} • ⏱️ {article.get('readingTime', 2)} min read</div>", 
                    unsafe_allow_html=True
                )
                
                # Description
                if 'description' in article and article['description']:
                    st.markdown(f"<div class='article-desc'>{article['description']}</div>", unsafe_allow_html=True)
                
                # Action buttons
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"<a href='{article['url']}' target='_blank' class='read-more'>Read Full Article</a>", unsafe_allow_html=True)
                with col2:
                    like = st.checkbox("❤️ Like", key=f"like_{idx}", value=article['title'] in st.session_state.get('liked_articles', []))
                    if like and article['title'] not in st.session_state.get('liked_articles', []):
                        if 'liked_articles' not in st.session_state:
                            st.session_state.liked_articles = []
                        st.session_state.liked_articles.append(article['title'])
                    elif not like and article['title'] in st.session_state.get('liked_articles', []):
                        st.session_state.liked_articles.remove(article['title'])
                
                st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying article: {str(e)}")

def display_trending_topics(df):
    """Display trending topics based on article frequency"""
    if df.empty:
        return
    
    try:
        # Extract keywords from titles (simplified approach)
        all_words = ' '.join(df['title'].fillna('')).lower()
        for word in ['the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are']:
            all_words = all_words.replace(f' {word} ', ' ')
        
        words = [word for word in all_words.split() if len(word) > 3]
        word_counts = pd.Series(words).value_counts().head(10)
        
        if word_counts.empty:
            st.info("Not enough data to display trending topics.")
            return
        
        # Create horizontal bar chart
        fig = px.bar(
            x=word_counts.values,
            y=word_counts.index,
            orientation='h',
            labels={'x': 'Frequency', 'y': 'Topic'},
            title='Trending Topics',
            color=word_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying trending topics: {str(e)}")

def display_sentiment_distribution(df):
    """Display sentiment distribution of articles"""
    if df.empty or 'sentiment' not in df.columns:
        return
    
    try:
        # Create histogram of sentiment scores
        fig = px.histogram(
            df,
            x='sentiment',
            nbins=20,
            labels={'sentiment': 'Sentiment Score', 'count': 'Number of Articles'},
            title='Article Sentiment Distribution',
            color_discrete_sequence=['#1E88E5']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying sentiment distribution: {str(e)}")

def display_publication_timeline(df):
    """Display publication timeline"""
    if df.empty or 'publishedDate' not in df.columns:
        return
    
    try:
        # Group by date and count articles
        timeline = df.groupby('publishedDate').size().reset_index(name='count')
        timeline = timeline.sort_values('publishedDate')
        
        if timeline.empty:
            st.info("Not enough data to display publication timeline.")
            return
        
        # Create line chart
        fig = px.line(
            timeline,
            x='publishedDate',
            y='count',
            labels={'publishedDate': 'Date', 'count': 'Number of Articles'},
            title='Publication Timeline',
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying publication timeline: {str(e)}")

def test_api_key(api_key):
    """Test if the API key is valid"""
    try:
        params = {
            "q": "test",
            "apiKey": api_key
        }
        response = requests.get(EVERYTHING_URL, params=params)
        data = response.json()
        return data.get("status") == "ok"
    except:
        return False

# --- INITIALIZE SESSION STATE ---
if 'liked_articles' not in st.session_state:
    st.session_state.liked_articles = []

if 'view_history' not in st.session_state:
    st.session_state.view_history = []

if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'favorite_categories': [],
        'excluded_sources': [],
        'reading_time_pref': 'any'
    }

if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = ""

# --- HEADER ---
st.markdown("<h1 class='main-header'>📰 NewsInsight Pro</h1>", unsafe_allow_html=True)
st.markdown("Your personalized news discovery and recommendation platform")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='sidebar-header'>📊 News Filters</div>", unsafe_allow_html=True)
    
    # API Key input
    st.markdown("### API Key")
    api_key_input = st.text_input(
        "Enter your NewsAPI key", 
        value=st.session_state.user_api_key,
        type="password",
        help="Get your free API key from newsapi.org"
    )
    
    if api_key_input != st.session_state.user_api_key:
        st.session_state.user_api_key = api_key_input
        if api_key_input:
            if test_api_key(api_key_input):
                st.success("API key is valid!")
            else:
                st.error("Invalid API key. Please check and try again.")
    
    # User profile (simulated)
    if st.session_state.liked_articles:
        st.info(f"You've liked {len(st.session_state.liked_articles)} articles")
    
    # Main filters
    main_category = st.selectbox("Main Category", MAIN_CATEGORIES, index=2)
    
    # Sub-category multi-select
    sub_categories = st.multiselect(
        "Sub-Categories", 
        SUB_CATEGORIES[main_category],
        default=[]
    )
    
    # Country selection
    country = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    country_code = COUNTRIES[country]
    
    # Date range
    st.markdown("##### Time Range")
    days_range = st.slider("Days to look back", 1, 30, 7)
    
    # Reading time preference
    reading_time = st.radio(
        "Reading Time Preference",
        ["Any", "Short (< 3 min)", "Medium (3-7 min)", "Long (> 7 min)"],
        index=0
    )
    
    # Search
    search_query = st.text_input("🔍 Search News", "")
    
    # Advanced filters expander
    with st.expander("Advanced Filters"):
        exclude_sources = st.multiselect(
            "Exclude Sources",
            ["CNN", "BBC News", "Fox News", "The Guardian", "Reuters"],
            default=[]
        )
        
        sort_by = st.radio(
            "Sort By",
            ["Relevance", "Newest First", "Popularity"],
            index=0
        )
        
        sentiment_filter = st.select_slider(
            "Sentiment Filter",
            options=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
            value="Neutral"
        )
    
    # Refresh button
    col1, col2 = st.columns(2)
    with col1:
        refresh = st.button("🔄 Refresh News", use_container_width=True)
    with col2:
        clear_likes = st.button("❌ Clear Likes", use_container_width=True)
        if clear_likes:
            st.session_state.liked_articles = []
            st.experimental_rerun()

# --- FETCH OR REFRESH NEWS ---
current_api_key = st.session_state.user_api_key if st.session_state.user_api_key else API_KEY
fetch_key = f"{main_category}_{country_code}_{search_query}_{days_range}_{current_api_key}"

if fetch_key not in st.session_state or refresh:
    with st.spinner("Fetching latest news..."):
        st.session_state[fetch_key] = fetch_news(
            category=main_category,
            country=country_code,
            query=search_query if search_query else None,
            days=days_range,
            api_key=current_api_key
        )

df = st.session_state[fetch_key]

# Apply filters to the dataframe
if df.empty:
    st.warning("No articles found matching your criteria. Try adjusting your filters or check your API key.")
    
    # Show API key guidance
    if current_api_key == DEFAULT_API_KEY:
        st.info("""
        ### 🔑 API Key Required for Full Access
        
        The default API key has limited access. For better results, especially for international news:
        
        1. Get your free API key from [NewsAPI.org](https://newsapi.org/register)
        2. Enter your API key in the sidebar
        3. Refresh the news
        
        With your own API key, you'll get access to more articles and international news sources.
        """)
else:
    try:
        # Apply reading time filter
        if reading_time != "Any":
            if reading_time == "Short (< 3 min)":
                df = df[df['readingTime'] < 3]
            elif reading_time == "Medium (3-7 min)":
                df = df[(df['readingTime'] >= 3) & (df['readingTime'] <= 7)]
            else:  # Long
                df = df[df['readingTime'] > 7]
        
        # Apply source exclusion
        if exclude_sources:
            df = df[~df['source'].apply(lambda x: x.get('name', '') in exclude_sources)]
        
        # Apply sorting
        if sort_by == "Newest First":
            df = df.sort_values('publishedAt', ascending=False)
        elif sort_by == "Popularity":
            df = df.sort_values('popularity', ascending=False)
        
        # Apply sentiment filter
        sentiment_map = {
            "Very Negative": (-1.0, -0.6),
            "Negative": (-0.6, -0.2),
            "Neutral": (-0.2, 0.2),
            "Positive": (0.2, 0.6),
            "Very Positive": (0.6, 1.0)
        }
        if sentiment_filter != "Neutral":
            sentiment_range = sentiment_map[sentiment_filter]
            df = df[(df['sentiment'] >= sentiment_range[0]) & (df['sentiment'] <= sentiment_range[1])]
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")

# --- MAIN CONTENT ---
# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📰 Top News", "🔍 Search Results", "❤️ Recommendations", "📊 Insights"])

with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("No articles found. Try adjusting your filters or search query.")
    else:
        try:
            # Display stats cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"<div class='stats-card'><div class='stats-number'>{len(df)}</div><div class='stats-label'>Articles</div></div>",
                    unsafe_allow_html=True
                )
            with col2:
                avg_sentiment = df['sentiment'].mean()
                sentiment_text = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
                st.markdown(
                    f"<div class='stats-card'><div class='stats-number'>{sentiment_text}</div><div class='stats-label'>Overall Sentiment</div></div>",
                    unsafe_allow_html=True
                )
            with col3:
                sources_count = df['source'].apply(lambda x: x.get('name', '')).nunique()
                st.markdown(
                    f"<div class='stats-card'><div class='stats-number'>{sources_count}</div><div class='stats-label'>News Sources</div></div>",
                    unsafe_allow_html=True
                )
            with col4:
                avg_reading = int(df['readingTime'].mean())
                st.markdown(
                    f"<div class='stats-card'><div class='stats-number'>{avg_reading}</div><div class='stats-label'>Avg. Reading Time</div></div>",
                    unsafe_allow_html=True
                )
            
            # Display articles in a grid
            st.subheader(f"Top {main_category.capitalize()} News from {country}")
            
            # Display articles in rows of 3
            for i in range(0, min(len(df), 15), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(df):
                        display_article_card(df.iloc[i + j], i + j, cols[j])
        except Exception as e:
            st.error(f"Error displaying top news: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    # Search interface
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query_tab = st.text_input("Search for specific news", value=search_query)
    with search_col2:
        search_button = st.button("Search", use_container_width=True)
    
    if search_button or search_query_tab:
        with st.spinner("Searching news..."):
            search_results = fetch_news(
                category=main_category,
                country=country_code,
                query=search_query_tab,
                days=days_range,
                api_key=current_api_key
            )
        
        if search_results.empty:
            st.info("No results found for your search query.")
        else:
            try:
                st.subheader(f"Search Results for '{search_query_tab}'")
                
                # Display search results
                for i in range(min(len(search_results), 10)):
                    display_article_card(search_results.iloc[i], f"search_{i}", st.container())
            except Exception as e:
                st.error(f"Error displaying search results: {str(e)}")
    else:
        st.info("Enter a search term above to find specific news articles.")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    if not st.session_state.liked_articles:
        st.info("Like some articles to get personalized recommendations!")
    else:
        try:
            st.subheader("Your Liked Articles")
            
            # Display liked articles
            liked_df = df[df['title'].isin(st.session_state.liked_articles)]
            if liked_df.empty:
                st.info("Your liked articles aren't in the current results. Try refreshing or changing filters.")
            else:
                for i, row in enumerate(liked_df.iterrows()):
                    display_article_card(row[1], f"liked_{i}", st.container(), show_image=False)
            
            # Get and display recommendations
            st.subheader("Recommended For You")
            recommendations = get_recommendations(df, st.session_state.liked_articles)
            
            if recommendations.empty:
                st.info("We couldn't generate recommendations based on your likes. Try liking more diverse articles.")
            else:
                for i, row in enumerate(recommendations.iterrows()):
                    article = row[1]
                    with st.container():
                        st.markdown(f"<div class='recommendation-card'>", unsafe_allow_html=True)
                        
                        # Show similarity score
                        similarity_pct = int(article['similarity'] * 100)
                        st.markdown(f"<span class='category-badge'>{similarity_pct}% Match</span>", unsafe_allow_html=True)
                        
                        # Article title and metadata
                        st.markdown(f"<div class='article-title'>{article['title']}</div>", unsafe_allow_html=True)
                        
                        # Source and date
                        published_time = article['publishedAt'].strftime('%b %d, %Y')
                        source_name = article.get('source', {}).get('name', 'Unknown')
                        
                        st.markdown(
                            f"<div class='article-meta'>📰 {source_name} • 🕒 {published_time}</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # Description
                        if 'description' in article and article['description']:
                            st.markdown(f"<div class='article-desc'>{article['description']}</div>", unsafe_allow_html=True)
                        
                        # Action buttons
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"<a href='{article['url']}' target='_blank' class='read-more'>Read Full Article</a>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying recommendations: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("No data available for insights. Try adjusting your filters.")
    else:
        try:
            st.subheader("News Insights & Analytics")
            
            # Display trending topics
            display_trending_topics(df)
            
            # Display sentiment and timeline in columns
            col1, col2 = st.columns(2)
            with col1:
                display_sentiment_distribution(df)
            with col2:
                display_publication_timeline(df)
            
            # Display source distribution
            sources = df['source'].apply(lambda x: x.get('name', 'Unknown')).value_counts().head(10)
            
            if not sources.empty:
                fig = px.pie(
                    values=sources.values,
                    names=sources.index,
                    title='Top News Sources',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display category word cloud (simulated)
            st.subheader("Popular Topics in This Category")
            
            # Create a simple tag cloud with most common words
            all_words = ' '.join(df['title'].fillna('')).lower()
            for word in ['the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are']:
                all_words = all_words.replace(f' {word} ', ' ')
        
            words = [word for word in all_words.split() if len(word) > 3]
            word_counts = pd.Series(words).value_counts().head(20)
            
            # Display as buttons
            st.write("Click on a topic to search for it:")
            buttons_html = ""
            for word, count in word_counts.items():
                size = min(1.5, max(0.8, count / word_counts.max() * 1.5))
                buttons_html += f'<a href="?search={word}" style="display:inline-block; margin:5px; font-size:{size}rem; padding:5px 10px; background-color:#E3F2FD; border-radius:15px; text-decoration:none; color:#1565C0;">{word} ({count})</a>'
            
            st.markdown(buttons_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying insights: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "NewsInsight Pro | Data refreshed every hour | Powered by NewsAPI",
    help="This app uses the NewsAPI to fetch real-time news data."
)

# Add a loading animation when switching tabs or refreshing
if refresh:
    with st.spinner("Refreshing data..."):
        time.sleep(1)  # Simulate loading

# Display API key information
st.sidebar.markdown("---")
st.sidebar.caption("Using NewsAPI for data")
if current_api_key == DEFAULT_API_KEY:
    st.sidebar.warning("""
    Using default API key with limited access.
    
    For full access to all features including international news:
    1. Get your free API key at [NewsAPI.org](https://newsapi.org)
    2. Enter it in the API Key field above
    """)

# Add debug information
with st.sidebar.expander("Debug Info", expanded=False):
    st.write(f"Articles fetched: {len(df) if not df.empty else 0}")
    st.write(f"API endpoint: {NEWSAPI_URL if not search_query else EVERYTHING_URL}")
    st.write(f"Category: {main_category}")
    st.write(f"Country: {country} ({country_code})")
    if not df.empty:
        st.write(f"Date range: {df['publishedDate'].min()} to {df['publishedDate'].max()}")