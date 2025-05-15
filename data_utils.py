import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

MAX_ROWS = 100_000

def load_and_cap(file):
    df = pd.read_csv(file, nrows=MAX_ROWS + 1)
    if df.empty:
        st.error("📂 The dataset is empty.")
        return None
    if len(df) > MAX_ROWS:
        st.warning(f"Dataset truncated to first {MAX_ROWS:,} rows for performance.")
        df = df.head(MAX_ROWS)
    return df

def map_columns(df):
    st.subheader("Column Mapping")
    cols = st.columns(2)
    with cols[0]:
        date_col = st.selectbox("Select Date column", df.columns)
    with cols[1]:
        num_cols = df.select_dtypes(include=np.number).columns
        price_col = st.selectbox("Select Price column", num_cols)

    if st.button("Confirm Columns"):
        if date_col == price_col:
            st.error("Date and Price columns cannot be the same")
            return None

        df = df[[date_col, price_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df = df.set_index("Date")
        return df

    return None

def add_technical_indicators(df):
    df = df.copy()
    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA14'] = df['Price'].rolling(window=14).mean()
    delta = df['Price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Returns'] = df['Price'].pct_change()
    return df

def plot_price(df):
    fig = px.line(df, y='Price', title="Price Over Time", labels={'Price':'Price (USD)'})
    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_distribution(df):
    fig = px.histogram(df, x='Price', nbins=50, title="Price Distribution")
    return fig

def plot_correlation(df):
    numeric_df = df.select_dtypes(include=np.number).dropna()
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        return fig
    return None

@st.cache_data
def cached_preprocess(df, time_step, test_split):
    df = df.dropna()
    scaler = MinMaxScaler((0,1))
    prices = df[['Price']].values
    scaled = scaler.fit_transform(prices)

    split_idx = int(len(scaled) * (1 - test_split))
    train, test = scaled[:split_idx], scaled[split_idx:]

    def make_seq(data):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i,0]); y.append(data[i,0])
        return np.array(X), np.array(y)

    X_tr, y_tr = make_seq(train)
    X_te, y_te = make_seq(test)

    X_tr = X_tr.reshape(-1, time_step, 1)
    X_te = X_te.reshape(-1, time_step, 1) if len(X_te) else np.empty((0, time_step,1))
    y_te = y_te if len(X_te) else np.empty((0,))

    return X_tr, y_tr, X_te, y_te, scaler
