# ================================================
# IHSG FORECAST STREAMLIT APP — FINAL OFFLINE + AUTO-FIX VERSION
# ================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, datetime as dt
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# --------------------------------
# CONFIG
# --------------------------------
st.set_page_config(page_title="📈 Prediksi IHSG Interaktif", layout="wide")
ARTIFACTS_DIR = "artifacts"
LOCAL_CSV_PATH = os.path.join(ARTIFACTS_DIR, "data_ihsg.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_xgb.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

# --------------------------------
# HELPER FUNCTIONS
# --------------------------------
def _find_close_col(df: pd.DataFrame):
    """Temukan kolom yang mengandung 'close'."""
    for col in df.columns:
        if "close" in str(col).lower():
            return col
    return None

def _ensure_1d_series(s):
    """Pastikan kolom menjadi Series 1D."""
    if isinstance(s, pd.DataFrame):
        return s.squeeze()
    return s

# --------------------------------
# FEATURE ENGINEERING
# --------------------------------
def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close_col = _find_close_col(df)
    if close_col is None:
        raise KeyError("Kolom 'close' tidak ditemukan di data input.")

    df["close"] = _ensure_1d_series(df[close_col]).astype(float)
    df["ret"] = df["close"].pct_change()
    df["logret"] = np.log1p(df["ret"])

    # Lags & rolling
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"lag_ret_{lag}"] = df["ret"].shift(lag)
        df[f"lag_close_{lag}"] = df["close"].shift(lag)
    for win in [5, 10, 20, 60]:
        df[f"roll_mean_{win}"] = df["close"].rolling(win).mean()
        df[f"roll_std_{win}"] = df["close"].rolling(win).std()
        df[f"roll_min_{win}"] = df["close"].rolling(win).min()
        df[f"roll_max_{win}"] = df["close"].rolling(win).max()

    # Technical Indicators
    try:
        rsi = RSIIndicator(df["close"])
        df["rsi14"] = rsi.rsi()
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        bb = BollingerBands(df["close"])
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])
    except Exception:
        pass

    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["target_ret_1d"] = df["ret"].shift(-1)
    df = df.dropna()
    return df


# --------------------------------
# FETCH DATA
# --------------------------------
@st.cache_data(show_spinner=False)
def fetch_data(start, end):
    """Ambil data IHSG dari CSV lokal (multiheader) atau fallback ke Stooq."""
    if os.path.exists(LOCAL_CSV_PATH):
        st.info("💾 Menggunakan data lokal IHSG (data_ihsg.csv)")
        try:
            # Coba baca multiheader (sesuai struktur file kamu)
            df = pd.read_csv(LOCAL_CSV_PATH, header=[0, 1])
            df.columns = [
                "_".join([str(x) for x in col if x and str(x) != "nan"]).strip().lower()
                for col in df.columns.values
            ]
            date_col = next((c for c in df.columns if "date" in c), None)
            if date_col is None:
                st.error("❌ File CSV tidak memiliki kolom 'Date' atau 'date'.")
                st.stop()

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
            df = df.loc[
                (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            ]
            if not df.empty:
                df.index.name = "date"
                return df, "Local CSV (MultiHeader)"
        except Exception as e:
            st.warning(f"⚠️ Gagal membaca file multiheader: {e}")

    # fallback ke Stooq
    try:
        from pandas_datareader import data as web
        st.warning("⚠️ File lokal tidak ditemukan, mengambil data dari Stooq...")
        df = web.DataReader("JKSE", "stooq", start, end)
        if not df.empty:
            df = df.rename(columns=str.lower).sort_index()
            df.index.name = "date"
            return df, "Stooq (JKSE)"
    except Exception as e:
        st.error(f"❌ Gagal ambil data dari Stooq: {e}")

    return pd.DataFrame(), None


# --------------------------------
# LOAD MODEL & META
# --------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH))
    return model, meta


# --------------------------------
# FORECAST FUNCTION
# --------------------------------
def forecast_prices(df, model, horizon_days, features):
    df = df.copy()
    preds, future_dates, future_prices = [], [], []

    close_col = _find_close_col(df)
    last_close = float(_ensure_1d_series(df[close_col]).iloc[-1])
    last_date = df.index[-1]

    for i in range(1, horizon_days + 1):
        new_row = df.iloc[-1:].copy()
        new_row.index = [last_date + dt.timedelta(days=i)]
        df = pd.concat([df, new_row])
        tmp = build_features(df)
        X_latest = tmp.reindex(columns=features, fill_value=0).iloc[-1:].values
        if X_latest.shape[0] == 0:
            continue
        pred_ret = model.predict(X_latest)[0]
        preds.append(pred_ret)
        future_dates.append(tmp.index[-1])
        last_close *= (1 + pred_ret)
        df.loc[df.index[-1], close_col] = last_close
        future_prices.append(last_close)

    return pd.DataFrame({"date": future_dates, "pred_ret": preds, "pred_price": future_prices})


# --------------------------------
# UI START
# --------------------------------
st.title("📈 Prediksi IHSG — Multi-Horizon Forecast (Interaktif)")
st.markdown("""
Model ini memprediksi **harga IHSG** untuk berbagai horizon waktu 
menggunakan **machine learning XGBoost** dengan data historis IHSG (offline mode).
""")

st.sidebar.header("⚙️ Pengaturan Data")
start = st.sidebar.date_input("Tanggal awal", dt.date(2015, 1, 1))
end = st.sidebar.date_input("Tanggal akhir", dt.date.today())
horizon_options = {
    "1 Hari": 1,
    "1 Minggu (5 Hari)": 5,
    "1 Bulan (~22 Hari)": 22,
    "3 Bulan (~66 Hari)": 66,
    "6 Bulan (~126 Hari)": 126,
    "1 Tahun (~252 Hari)": 252
}
horizon_label = st.sidebar.selectbox("Horizon Prediksi", list(horizon_options.keys()))
horizon_days = horizon_options[horizon_label]

# Load model
model, meta = load_model()

# Fetch data
with st.spinner("📥 Mengambil data IHSG..."):
    df, source = fetch_data(start, end)
    if df.empty:
        st.error("❌ Data IHSG kosong! Tidak ada data untuk periode yang dipilih.")
        st.stop()

# Build features
df = build_features(df)

# Forecast
with st.spinner(f"🔮 Membuat prediksi {horizon_label} ke depan..."):
    forecast_df = forecast_prices(df, model, horizon_days, meta["features"])

# Confidence interval
pred_std = np.std(forecast_df["pred_ret"])
forecast_df["upper"] = forecast_df["pred_price"] * (1 + pred_std)
forecast_df["lower"] = forecast_df["pred_price"] * (1 - pred_std)

# Combine
hist_close_col = _find_close_col(df)
combined_hist = pd.DataFrame({"date": df.index, "price": _ensure_1d_series(df[hist_close_col])})
combined_fore = pd.DataFrame({"date": forecast_df["date"], "price": forecast_df["pred_price"]})

# Plot
st.subheader(f"📊 Grafik Harga IHSG + Prediksi ({horizon_label})")
fig = go.Figure()
fig.add_trace(go.Scatter(x=combined_hist["date"], y=combined_hist["price"], mode="lines", name="Harga Historis", line=dict(color="blue", width=2)))
fig.add_trace(go.Scatter(x=combined_fore["date"], y=combined_fore["price"], mode="lines", name="Prediksi Harga", line=dict(color="orange", width=3, dash="dash")))
fig.add_trace(go.Scatter(x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
                         y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
                         fill="toself", fillcolor="rgba(255,165,0,0.2)", line=dict(color="rgba(255,255,255,0)"),
                         showlegend=True, name="Confidence ±σ"))
fig.update_layout(xaxis_title="Tanggal", yaxis_title="Harga IHSG", template="plotly_dark",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# Metric
predicted_price = forecast_df["pred_price"].iloc[-1]
last_price = _ensure_1d_series(df[hist_close_col]).iloc[-1]
change_pct = ((predicted_price - last_price) / last_price) * 100
st.metric(f"Perkiraan Harga IHSG ({horizon_label} ke depan)", f"{predicted_price:,.0f}", f"{change_pct:+.2f}%")

# Table
st.subheader("📅 Detail Hasil Prediksi per Hari")
forecast_df_display = forecast_df.copy()
forecast_df_display["date"] = forecast_df_display["date"].dt.strftime("%Y-%m-%d")
forecast_df_display["pred_ret"] = forecast_df_display["pred_ret"] * 100
forecast_df_display = forecast_df_display.rename(columns={
    "date": "Tanggal", "pred_price": "Prediksi Harga (Rp)", "pred_ret": "Prediksi Return (%)"})
st.dataframe(forecast_df_display[["Tanggal", "Prediksi Harga (Rp)", "Prediksi Return (%)"]]
             .style.format({"Prediksi Harga (Rp)": "{:,.0f}", "Prediksi Return (%)": "{:+.4f}"}))

csv_data = forecast_df_display.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download hasil prediksi (CSV)", data=csv_data,
                   file_name=f"IHSG_Forecast_{horizon_label.replace(' ', '_')}.csv",
                   mime="text/csv")

st.caption("📘 Catatan: Prediksi bersifat indikatif dan tidak menjamin hasil investasi.")
