import streamlit as st
import plotly.express as px
import pandas as pd
import data_utils
import model_utils
from io import BytesIO

MAX_ROWS = 100_000

load_and_cap = data_utils.load_and_cap
preprocess = data_utils.cached_preprocess

st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.title("📈 Bitcoin Price Prediction + Forecaster")
st.write("""
Upload a Bitcoin CSV, map **Date** and **Price** columns, pick a date range & frequency,
then click **Train / Evaluate** to fit an LSTM model, view metrics, forecast future prices,
and download results.
""")

uploaded_file = st.file_uploader("Upload Bitcoin price CSV", type=["csv"])
if not uploaded_file:
    st.stop()

raw_df = load_and_cap(uploaded_file)
if raw_df is None or raw_df.empty:
    st.error("Unable to load data or dataset is empty.")
    st.stop()

if "mapped" not in st.session_state:
    df = data_utils.map_columns(raw_df)
    if df is None:
        st.stop()
    st.session_state.df = df
    st.session_state.mapped = True

df = st.session_state.df
df = df[~df.index.isna()]
if df.empty:
    st.error("No valid timestamps after cleaning.")
    st.stop()

min_date, max_date = df.index.min().date(), df.index.max().date()
start, end = st.date_input(
    "Plot range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
freq = st.selectbox("Resample frequency", ["15T", "H", "D"])

df = df.loc[start:end]
if df.empty:
    st.error("No data in selected date range.")
    st.stop()

df = df.resample(freq).mean().dropna()
df = data_utils.add_technical_indicators(df)
df = df.dropna()

col1, col2 = st.columns(2)
with col1:
    time_step = st.slider("Look-back window (steps)", 5, 120, 60, 5)
with col2:
    test_split = st.slider("Test split (%)", 5, 40, 20, 5) / 100.0

st.success("✅ Data loaded and preprocessed")
st.write(df.head())

tab1, tab2, tab3 = st.tabs(["Price Trend", "Distribution", "Correlation"])
with tab1:
    st.plotly_chart(data_utils.plot_price(df), use_container_width=True)
with tab2:
    st.plotly_chart(data_utils.plot_distribution(df), use_container_width=True)
with tab3:
    corr_fig = data_utils.plot_correlation(df)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

train_disabled = len(df) < time_step * 2
if train_disabled:
    st.info("Increase the date range or reduce look-back window to enable training.")

model = None
scaler = None

if st.button("🚀 Train / Evaluate", disabled=train_disabled):
    with st.spinner("Training & evaluating..."):
        X_tr, y_tr, X_te, y_te, scaler = preprocess(df, time_step, test_split)
        st.write(f"Train sequences: {X_tr.shape[0]} | Test sequences: {X_te.shape[0]}")

        if X_te.size > 0:
            X_val, y_val = X_te, y_te
        else:
            st.warning("⚠️ No test data available. Using training set for validation. Metrics may be unreliable.")
            X_val, y_val = X_tr, y_tr

        model, preds, history = model_utils.train_model(
            X_tr, y_tr, X_val, y_val
        )

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.df = df
        st.session_state.time_step = time_step
        st.session_state.freq = freq

        # Show model summary
        with st.expander("Model Summary"):
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            summary = "\n".join(stringlist)
            st.text(summary)

        model_utils.plot_loss(history)

        if X_te.size > 0:
            rmse, mae, y_true, y_pred = model_utils.evaluate_model(y_te, preds, scaler)
            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")

            st.subheader("Predicted vs Actual")
            plot_df = (
                df.iloc[-len(y_true):]
                .assign(Actual=y_true.flatten(), Predicted=y_pred.flatten())
            )

            fig = px.line(plot_df, y=["Actual", "Predicted"], title="Predictions vs Actual Prices")
            st.plotly_chart(fig, use_container_width=True)

            # Download predictions button
            csv = plot_df[['Actual', 'Predicted']].to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
        else:
            st.warning("⚠️ No test set available for evaluation. Model was trained but results not plotted.")


if 'model' in st.session_state and 'scaler' in st.session_state:
    if st.button("📈 Forecast Future Prices"):
        n_future = st.slider("How many future steps to predict?", 1, 100, 10)
        future_df = model_utils.forecast_future(
            st.session_state.model,
            st.session_state.df,
            st.session_state.scaler,
            st.session_state.time_step,
            n_future,
            st.session_state.freq
        )

        st.subheader("Future Forecast")
        st.line_chart(future_df['Predicted Price'])

        # Download button
        csv_future = future_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Future Predictions CSV",
            data=csv_future,
            file_name='future_forecast.csv',
            mime='text/csv'
        )
