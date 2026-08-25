import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Config Halaman Dashboard
st.set_page_config(page_title="EWS Ketahanan Pangan Sumatera", layout="wide")

st.title("🐟 Dashboard Early Warning System (EWS) Ikan Gabus Sumatera")
st.write("Sistem Pemantauan Pasokan Protein Lokal & Prediksi Inflasi Berbasis Machine Learning")

# 1. LOAD DATASET
@st.cache_data
def load_data():
    df = pd.read_excel("Dataset Sumatera Ikan Gabus.xlsx")
    df['Provinsi'] = df['Provinsi'].replace({'KEP. BANGKA BELITUNG': 'KEPULAUAN BANGKA BELITUNG'})
    prov_df = df.groupby(['Provinsi', 'Tahun']).agg({
        'Volume (ton)': 'sum',
        'Nilai (Rp. Juta)': 'sum'
    }).reset_index()
    prov_df['Harga_Rp_kg'] = ((prov_df['Nilai (Rp. Juta)'] * 1000000) / (prov_df['Volume (ton)'] * 1000)).round(2)
    return prov_df.sort_values(['Provinsi', 'Tahun']).reset_index(drop=True)

df_prov = load_data()

# 2. MACHINE LEARNING ENGINE
df_prov['Volume_Lag1'] = df_prov.groupby('Provinsi')['Volume (ton)'].shift(1)
df_prov['Harga_Lag1'] = df_prov.groupby('Provinsi')['Harga_Rp_kg'].shift(1)
ml_df = df_prov.dropna().copy()

X = ml_df[['Volume_Lag1', 'Harga_Lag1', 'Nilai (Rp. Juta)']]
y = ml_df['Volume (ton)']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
ml_df['Prediksi_Volume'] = model.predict(X)

def get_status(row):
    pct = ((row['Prediksi_Volume'] - row['Volume_Lag1']) / row['Volume_Lag1']) * 100
    if pct <= -10:
        return '🔴 AWAS (Risiko Inflasi)'
    elif pct < 0:
        return '🟡 WASPADA'
    else:
        return '🟢 AMAN (Surplus)'

ml_df['Status_EWS'] = ml_df.apply(get_status, axis=1)

# 3. INTERFACE STREAMLIT
sidebar_prov = st.sidebar.selectbox("Pilih Provinsi:", ml_df['Provinsi'].unique())
data_selected = ml_df[ml_df['Provinsi'] == sidebar_prov]
latest_data = data_selected.iloc[-1]

# Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Volume Produksi Terakhir (Ton)", f"{latest_data['Volume (ton)']:,.2f}")
col2.metric("Prediksi Volume ML (Ton)", f"{latest_data['Prediksi_Volume']:,.2f}")
col3.subheader(f"Status: {latest_data['Status_EWS']}")

st.divider()

# Grafik Tren Produksi vs Prediksi
fig = px.line(data_selected, x='Tahun', y=['Volume (ton)', 'Prediksi_Volume'], 
              title=f"Tren Produksi Real vs Prediksi ML - {sidebar_prov}",
              labels={'value': 'Volume (Ton)', 'variable': 'Kategori'},
              markers=True)
st.plotly_chart(fig, use_container_width=True)

# Tabel EWS Keseluruhan Sumatera
st.subheader("📌 Ringkasan Status EWS Seluruh Provinsi (2024)")
st.dataframe(ml_df[ml_df['Tahun'] == 2024][['Provinsi', 'Volume (ton)', 'Prediksi_Volume', 'Harga_Rp_kg', 'Status_EWS']], use_container_width=True)