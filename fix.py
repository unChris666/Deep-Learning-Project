import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from datetime import timedelta
import os

# --- 1. CONFIGURATION ---
# st.set_page_config(page_title="LPG Demand Intelligence Hub", layout="wide", page_icon="🔋")
st.set_page_config(page_title="LPG Demand Intelligence Hub", layout="wide")

# Sembunyikan warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# KONFIGURASI WINDOW SIZE (Wajib sama dengan saat training)
WINDOW_SIZE = 3 

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Load 3 Models
    model_rnn = tf.keras.models.load_model('rnn_model.h5', compile=False)
    model_lstm = tf.keras.models.load_model('lstm_model.h5', compile=False)
    model_gru = tf.keras.models.load_model('gru_model.h5', compile=False)
    
    # Load Scaler
    scaler = joblib.load('scaler.pkl')
    
    # Load Data & Preprocessing
    df = pd.read_csv('Data Penjualan Gas - Data Final.csv')
    df['date'] = pd.to_datetime(df['Tanggal '], format='%d/%m/%Y')
    df = df.rename(columns={'Total Penjualan': 'demand'})
    df = df[['date', 'demand']]
    df.set_index('date', inplace=True)
    
    # Resample 3 Hari
    df_3day = df.resample('3D').sum()
    
    return model_rnn, model_lstm, model_gru, scaler, df_3day

try:
    model_rnn, model_lstm, model_gru, scaler, df_hist = load_assets()
except Exception as e:
    st.error(f"Gagal memuat aset: {e}")
    st.stop()

# --- 3. RECURSIVE PREDICTION FUNCTION ---
def predict_until_target(model, scaler, df_history, window_size, target_date):
    """
    Fungsi Generik untuk prediksi rekursif.
    """
    current_sequence = df_history['demand'].values[-window_size:].tolist()
    current_date = df_history.index[-1]
    
    future_predictions = []
    future_dates = []
    
    while current_date < target_date:
        input_seq = np.array(current_sequence[-window_size:]).reshape(-1, 1)
        input_scaled = scaler.transform(input_seq)
        input_model = input_scaled.reshape(1, window_size, 1)
        
        pred_scaled = model.predict(input_model, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]
        pred_value = max(0, int(pred_value)) 
        
        current_date = current_date + timedelta(days=3)
        
        future_predictions.append(pred_value)
        future_dates.append(current_date)
        current_sequence.append(pred_value)
    
    df_future = pd.DataFrame({
        'date': future_dates,
        'demand': future_predictions
    })
    df_future.set_index('date', inplace=True)
    return df_future

# --- 4. DASHBOARD UI LAYOUT ---

st.title("🔋 LPG Demand Intelligence Hub")
st.markdown("*Multi-Model Comparison: RNN vs LSTM vs GRU*")
st.divider()

# Layout Utama: 30% Kiri (Control), 70% Kanan (Visual)
col_control, col_main = st.columns([3, 7], gap="medium")

# --- BAGIAN KIRI: CONTROL PANEL (30%) ---
with col_control:
    with st.container(border=True):
        st.subheader("Control Panel")
        
        # 1. Info Data Terakhir (Baris 1)
        last_hist_date = df_hist.index[-1]
        st.info(f"**Data Terakhir:**\n\n{last_hist_date.strftime('%d %B %Y')}")
        
        st.write("---") # Separator visual
        
        # 2. Input Tanggal (Baris 2)
        min_date = last_hist_date + timedelta(days=1)
        max_date = last_hist_date + timedelta(days=30) 
        
        target_date_input = st.date_input(
            "**Target Tanggal Prediksi:**",
            value=min_date + timedelta(days=3), 
            min_value=min_date,
            max_value=max_date,
            help="Pilih tanggal di masa depan (Maks 30 hari)."
        )
        target_date = pd.to_datetime(target_date_input)
        
        st.write("---") # Separator visual

        # 3. Tombol Eksekusi (Baris 3)
        run_prediction = st.button("PREDIKSI", type="primary", use_container_width=True)

# --- BAGIAN KANAN: PREVIEW & HASIL (70%) ---
with col_main:
    
    # KONDISI 1: SEBELUM TOMBOL DITEKAN (Preview Mode)
    if not run_prediction:
        st.subheader("Preview Data Historis")
        
        # Statistik Deskriptif Sederhana
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data Point", f"{len(df_hist)} Periode")
        c2.metric("Rata-rata Permintaan", f"{int(df_hist['demand'].mean())} Tabung")
        c3.metric("Permintaan Tertinggi", f"{int(df_hist['demand'].max())} Tabung")
        
        # Plot Data Historis Full
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df_hist.index, y=df_hist['demand'],
            mode='lines', name='Data Historis',
            line=dict(color='#3366CC', width=2),
            fill='tozeroy', # Efek area di bawah grafik agar cantik
            fillcolor='rgba(51, 102, 204, 0.1)'
        ))
        fig_hist.update_layout(
            title="Grafik Permintaan Historis (Resampled 3-Day)",
            xaxis_title="Tanggal", yaxis_title="Jumlah Tabung",
            hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        with st.expander("Lihat Data Mentah (Tabel)"):
            st.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)

    # KONDISI 2: SETELAH TOMBOL DITEKAN (Result Mode)
    else:
        try:
            with st.spinner(f"Sedang mensimulasikan 3 model AI hingga {target_date.strftime('%d %b %Y')}..."):
                # Eksekusi Model
                df_rnn = predict_until_target(model_rnn, scaler, df_hist, WINDOW_SIZE, target_date)
                df_lstm = predict_until_target(model_lstm, scaler, df_hist, WINDOW_SIZE, target_date)
                df_gru = predict_until_target(model_gru, scaler, df_hist, WINDOW_SIZE, target_date)
            
            if df_rnn.empty:
                st.warning("Tanggal target terlalu dekat.")
            else:
                final_date_str = df_rnn.index[-1].strftime('%d %b %Y')

                # --- BAGIAN 1: HASIL PREDIKSI (Cards) ---
                st.subheader(f"Hasil Prediksi Target: {final_date_str}")
                
                # Container hasil dengan styling
                with st.container():
                    m1, m2, m3 = st.columns(3)
                    
                    val_rnn = df_rnn.iloc[-1]['demand']
                    val_lstm = df_lstm.iloc[-1]['demand']
                    val_gru = df_gru.iloc[-1]['demand']
                    
                    # Custom CSS styling logic could go here, but using standard metrics for stability
                    m1.metric("RNN Prediction", f"{val_rnn}", "Tabung", border=True)
                    m2.metric("LSTM Prediction", f"{val_lstm}", "Tabung", border=True)
                    m3.metric("GRU Prediction", f"{val_gru}", "Tabung", border=True)

                # Rata-rata Ensemble 
                avg_pred = int((val_rnn + val_lstm + val_gru) / 3)
                st.caption(f"**Insight**: Rata-rata prediksi dari ketiga model adalah **{avg_pred}** Tabung.")
                # st.divider()

                # --- BAGIAN 2: TRAJEKTORI MULTI-MODEL (Chart) ---
                st.subheader("Trajektori Prediksi Multi-Model")
                
                fig = go.Figure()
                
                # A. Data Historis (Fokus 10 titik terakhir)
                recent_hist = df_hist.tail(10)
                last_hist_date = recent_hist.index[-1]
                last_hist_val = recent_hist['demand'].iloc[-1]
                
                fig.add_trace(go.Scatter(
                    x=recent_hist.index, y=recent_hist['demand'],
                    mode='lines+markers', name='Data Asli',
                    line=dict(color='#3366CC', width=3)
                ))
                
                # Fungsi Helper untuk Gambar Garis Model & Konektornya
                def add_model_trace(fig, df_pred, name, color, dash_style):
                    # 1. Garis Konektor (Hist Akhir -> Prediksi Awal)
                    fig.add_trace(go.Scatter(
                        x=[last_hist_date, df_pred.index[0]],
                        y=[last_hist_val, df_pred['demand'].iloc[0]],
                        mode='lines', showlegend=False,
                        line=dict(color=color, dash=dash_style, width=1)
                    ))
                    # 2. Garis Prediksi Utama
                    fig.add_trace(go.Scatter(
                        x=df_pred.index, y=df_pred['demand'],
                        mode='lines+markers', name=name,
                        line=dict(color=color, dash=dash_style, width=2)
                    ))

                # B. Plot Model RNN (Orange)
                add_model_trace(fig, df_rnn, 'RNN', '#FF5733', 'dot')
                
                # C. Plot Model LSTM (Green)
                add_model_trace(fig, df_lstm, 'LSTM', '#33FF57', 'dash')
                
                # D. Plot Model GRU (Blue)
                add_model_trace(fig, df_gru, 'GRU', '#3357FF', 'longdash')
                
                fig.update_layout(
                    xaxis_title="Tanggal", 
                    yaxis_title="Jumlah Tabung",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.1),
                    height=450,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- BAGIAN 3: RINCIAN DATA (Tabs) ---
                st.subheader("📋 Rincian Data Tabular")
                tab1, tab2, tab3 = st.tabs(["RNN Results", "LSTM Results", "GRU Results"])
                
                with tab1: st.dataframe(df_rnn, use_container_width=True)
                with tab2: st.dataframe(df_lstm, use_container_width=True)
                with tab3: st.dataframe(df_gru, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.markdown("---")
            st.write("🔍 **Debug Info:**")
            st.write(f"- Window Size setting: {WINDOW_SIZE}")
            st.write(f"- Total Data History: {len(df_hist)}")

            st.write("- Pastikan scaler.pkl cocok dengan data demand.")
