import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from scipy.stats import zscore

# 1️⃣ Load & preprocess data (using file uploader)
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['datetime'] = pd.to_timedelta(df['step'], unit='h') + pd.Timestamp('2020-01-01')
    df['date'] = df['datetime'].dt.date
    return df

@st.cache_data
def aggregate_data(df):
    grouped = df.groupby(['date', 'type']).agg(
        total_amount=('amount', 'sum'),
        transaction_count=('amount', 'count')
    ).reset_index()
    return grouped

# 2️⃣ Forecasting function
def forecast_transaction(df, transaction_type, periods=30):
    df_type = df[df['type'] == transaction_type]
    df_prophet = df_type[['date', 'total_amount']].rename(columns={'date': 'ds', 'total_amount': 'y'}).dropna()

    if df_prophet.shape[0] < 2:
        return None, None

    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def forecast_all_transactions(df, types, periods=30):
    all_forecasts = pd.DataFrame()
    for t in types:
        model, forecast = forecast_transaction(df, t, periods)
        if forecast is not None:
            forecast['type'] = t
            all_forecasts = pd.concat([all_forecasts, forecast[['ds', 'yhat', 'type']]])
    return all_forecasts

# 3️⃣ Anomaly detection
def detect_anomalies(df, transaction_type, threshold=2.0):
    df_type = df[df['type'] == transaction_type].copy()
    df_type['zscore'] = zscore(df_type['total_amount'])
    df_type['anomaly'] = df_type['zscore'].apply(
        lambda x: 'Spike' if x > threshold else ('Drop' if x < -threshold else 'Normal')
    )
    return df_type

# 🔷 Streamlit UI
st.set_page_config(page_title="Transaction Forecast & Anomaly Detection", layout="wide")
st.title("📊 Transaction Forecast & Anomaly Detection")

# Upload or load file
uploaded_file = st.file_uploader("Upload your CSV (max 25MB)", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to continue.")

if uploaded_file:
    grouped = aggregate_data(df)
    types = grouped['type'].unique()

    # Sidebar filters
    st.sidebar.header("Options")
    selected_type = st.sidebar.selectbox("Transaction Type", types)
    forecast_period = st.sidebar.slider("Forecast period (days)", 7, 60, 30)

    # 🔮 Forecasting Section
    st.header(f"🔮 Forecast: {selected_type}")
    model, forecast = forecast_transaction(grouped, selected_type, forecast_period)

    if model is None:
        st.warning("Not enough data to forecast.")
    else:
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    # 🚨 Anomaly Detection
    st.header(f"🚨 Anomaly Detection: {selected_type}")
    anomaly_df = detect_anomalies(grouped, selected_type)

    colors = {'Spike': 'red', 'Drop': 'blue', 'Normal': 'gray'}
    fig3, ax = plt.subplots(figsize=(12, 5))
    for label, color in colors.items():
        subset = anomaly_df[anomaly_df['anomaly'] == label]
        ax.scatter(subset['date'], subset['total_amount'], label=label, color=color)
    ax.plot(anomaly_df['date'], anomaly_df['total_amount'], color='black', alpha=0.3)
    ax.set_title(f'{selected_type} – Anomaly Detection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Amount')
    ax.legend()
    st.pyplot(fig3)

    # 📊 All Forecasts Overlay
    if st.checkbox("Compare All Transaction Types"):
        st.header("📈 Forecast Comparison")
        all_forecasts = forecast_all_transactions(grouped, types, forecast_period)

        fig4, ax = plt.subplots(figsize=(14, 6))
        for t in all_forecasts['type'].unique():
            subset = all_forecasts[all_forecasts['type'] == t]
            ax.plot(subset['ds'], subset['yhat'], label=t)
        ax.set_title('Forecast Across All Transaction Types')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predicted Total Amount')
        ax.legend()
        st.pyplot(fig4)
