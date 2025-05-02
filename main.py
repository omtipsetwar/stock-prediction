import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import requests
from textblob import TextBlob

st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 8px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Stock Market Analysis")
st.sidebar.header("🔧 Stock Name")
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", "")

if not stock:
    st.sidebar.error("Please enter a valid stock ticker symbol.")
    st.stop()

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def fetch_stock_data(stock):
    try:
        data = yf.download(stock, start=start, end=end)
        if data.empty:
            st.error("No data found for the specified stock.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def fetch_stock_news(stock):
    api_key = "27bb78beb8644545890fc0f8e1012968"
    api_url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}"
    try:
        response = requests.get(api_url)
        data = response.json()
        if data["status"] == "ok":
            return data["articles"]
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(news_articles):
    sentiments = []
    for article in news_articles:
        text = (article.get("title") or "") + " " + (article.get("description") or "")
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        if score > 0:
            sentiment = "Positive"
        elif score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        sentiments.append((article["title"], article["url"], sentiment))
    return sentiments

def predict_stock_price(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    closing_price = stock_data['Close'].values.reshape(-1, 1)
    closing_price_scaled = scaler.fit_transform(closing_price)
    x_input = closing_price_scaled[-60:].reshape(1, -1, 1)
    prediction = model.predict(x_input)
    return scaler.inverse_transform(prediction)[0][0]

def generate_esg_data(length):
    df = pd.DataFrame({
        'Environmental Score': np.random.uniform(50, 100, size=length),
        'Social Score': np.random.uniform(50, 100, size=length),
        'Governance Score': np.random.uniform(50, 100, size=length)
    })
    df['Average Score'] = df[['Environmental Score', 'Social Score', 'Governance Score']].mean(axis=1)
    return df

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values, 'orange', label='Moving Average')
    ax.plot(full_data['Close'], 'blue', label='Close Price')
    if extra_data:
        ax.plot(extra_dataset, 'green', label='Extra Data')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price and Moving Averages")
    ax.legend()
    return fig

def get_currency_symbol(stock):
    if stock.lower().endswith((".ns", ".bs")):
        return "₹"
    return "$"

if st.sidebar.button("Live News"):
    stock_data = fetch_stock_data(stock)
    if stock_data is not None:
        st.subheader(f"📊 Stock Data for {stock}")
        st.dataframe(stock_data)

        news_articles = fetch_stock_news(stock)
        if news_articles:
            st.subheader(f"📰 Latest News for {stock}")
            sentiments = analyze_sentiment(news_articles)
            for title, url, sentiment in sentiments:
                st.write(f"**{title}**")
                st.write(f"[Read more]({url})")
                st.write(f"Sentiment: {sentiment}")
                st.markdown("---")
        else:
            st.warning("No news found.")

        st.subheader("📊 Stock Price Chart")
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data['Close'], label=f'{stock} Close Price')
        plt.title(f'{stock} Stock Price')
        plt.xlabel("Date")
        plt.ylabel("Close Price (₹)")
        plt.legend()
        st.pyplot(plt)

if st.sidebar.button("Fetch Stock Data", key="fetch_stock_data_button_1"):
    stock_data = fetch_stock_data(stock)

    if stock_data is not None:
        st.subheader("📊 Stock Data Overview")
        st.dataframe(stock_data)

        st.subheader("📈 Stock Price Chart")
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data['Close'], label=f'{stock} Close Price')
        plt.title(f'{stock} Stock Price')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        st.pyplot(plt)

        esg_df = generate_esg_data(len(stock_data))
        st.subheader("♻️ ESG Scores")
        st.dataframe(esg_df)

        splitting_len = int(len(stock_data) * 0.7)
        x_test = stock_data[['Close']][splitting_len:]

        for ma_period in [50, 100, 200]:
            stock_data[f'MA_for_{ma_period}_days'] = stock_data['Close'].rolling(ma_period).mean()
            st.subheader(f'📈 {ma_period}-Day Moving Average')
            st.pyplot(plot_graph((15, 6), stock_data[f'MA_for_{ma_period}_days'], stock_data))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test)

        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])
        x_data, y_data = np.array(x_data), np.array(y_data)

        predictions = model.predict(x_data)
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        plotting_data = pd.DataFrame({
            'Original Test Data': inv_y_test.flatten(),
            'Predicted Test Data': inv_pre.flatten()
        }, index=stock_data.index[splitting_len + 100:])

        st.subheader("Original vs Predicted Values")
        st.dataframe(plotting_data)

        st.subheader('📈 Close Price Comparison')
        fig = plt.figure(figsize=(15, 6))
        plt.plot(pd.concat([stock_data['Close'][:splitting_len + 100], plotting_data], axis=0),
                 label='Original Close Price', color='blue')
        plt.plot(plotting_data['Predicted Test Data'], label='Predicted Close Price', color='orange')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Comparison of Original and Predicted Stock Prices")
        st.pyplot(fig)

        # 🎯 Investment Recommendation
        latest_close_price = float(stock_data['Close'].iloc[-1])
        ma_50 = float(stock_data['MA_for_50_days'].iloc[-1])
        ma_100 = float(stock_data['MA_for_100_days'].iloc[-1])
        ma_200 = float(stock_data['MA_for_200_days'].iloc[-1])
        avg_esg_score = esg_df['Average Score'].iloc[-1]
        currency_symbol = get_currency_symbol(stock)

        # Assign ESG level and color
        esg_label = "High ESG" if avg_esg_score >= 75 else "Moderate ESG"
        esg_color = "blue" if avg_esg_score >= 75 else "darkorange"

        # Recommendation and color logic
        if latest_close_price > ma_50 and latest_close_price > ma_100 and latest_close_price > ma_200:
            rec_text = "BUY"
            rec_color = "green"
        elif latest_close_price < ma_50 and latest_close_price < ma_100 and latest_close_price < ma_200:
            rec_text = "SELL"
            rec_color = "red"
        else:
            rec_text = "HOLD"
            rec_color = "orange"

        # 🎯 Display recommendation with badges and icons
st.markdown("### 📌 **Investment Recommendation**")
st.markdown(f"**Latest Close Price:** {currency_symbol}{latest_close_price:.2f}")
st.markdown(f"**50-Day Moving Average:** {currency_symbol}{ma_50:.2f}")
st.markdown(f"**100-Day Moving Average:** {currency_symbol}{ma_100:.2f}")
st.markdown(f"**200-Day Moving Average:** {currency_symbol}{ma_200:.2f}")
st.markdown(f"**Average ESG Score:** {avg_esg_score:.2f}")

# Icons and badge styles
icons = {
    "BUY": "🟢💹",
    "SELL": "🔴📉",
    "HOLD": "🟠🕒"
}

esg_icons = {
    "High ESG": "🌿✅",
    "Moderate ESG": "♻️⚠️"
}

rec_badge = f"""
<div style="display: inline-block; padding: 10px 20px; background-color:{rec_color}; 
            color: white; border-radius: 25px; font-weight: bold; font-size: 18px; margin-right: 10px;">
    {icons[rec_text]} {rec_text}
</div>
"""

esg_badge = f"""
<div style="display: inline-block; padding: 10px 20px; background-color:{esg_color}; 
            color: white; border-radius: 25px; font-weight: bold; font-size: 18px;">
    {esg_icons[esg_label]} {esg_label}
</div>
"""

st.markdown("#### 🏅 **Final Verdict:**", unsafe_allow_html=True)
st.markdown(f"{rec_badge} {esg_badge}", unsafe_allow_html=True)
