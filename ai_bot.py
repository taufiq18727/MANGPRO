import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GaussianNoise
from sklearn.preprocessing import MinMaxScaler

# --- 1. KONFIGURASI PELATIHAN ---
# Saham yang dijadikan bahan belajar (General Model)
tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", 
    "ASII.JK", "ANTM.JK", "MDKA.JK", "GOTO.JK",
    "ADRO.JK", "PGAS.JK", "UNVR.JK", "ICBP.JK"
]

SEQ_LEN = 60  # Melihat 60 hari ke belakang (approx 3 bulan data trading)

def add_features(df):
    df = df.copy()
    # RSI (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Weekly Trend (Harga hari ini vs 5 hari lalu)
    df['Weekly_Trend'] = df['Close'] / df['Close'].shift(5)
    
    # Volatility (Volume Change)
    df['Vol_Change'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df.dropna()

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 3):
        x = data[i:(i + seq_length)]
        # Target: Apakah harga 3 hari ke depan > harga hari ini? (Swing)
        current_price = data[i + seq_length - 1, 3] # Index 3 adalah Close
        future_price = data[i + seq_length + 2, 3]  
        
        y = 1 if future_price > current_price else 0
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train():
    print("🏋️‍♂️ Memulai Retraining Mingguan...")
    
    all_X, all_y = [], []
    processed_data_list = []
    
    # 1. Download & Preprocess Data
    # Ambil 2 tahun terakhir untuk memastikan cukup data buat deep learning
    print("⬇️ Download data terbaru...")
    raw_data = yf.download(tickers, period="2y", auto_adjust=True, group_by='ticker', threads=True)
    
    # 2. Siapkan Scaler
    # Kita pakai 1 Scaler untuk semua saham agar model paham "General Pattern"
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Kumpulkan semua data valid untuk fitting scaler
    temp_list_for_scaling = []
    
    for t in tickers:
        try:
            df = raw_data[t].copy()
            if len(df) < 200: continue
            
            df = add_features(df)
            
            # Kolom Fitur: Open, High, Low, Close, Volume, RSI, Weekly, VolChange
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'Weekly_Trend', 'Vol_Change']
            vals = df[cols].values
            
            temp_list_for_scaling.append(vals)
            processed_data_list.append(vals) # Simpan untuk sequence nanti
        except Exception as e:
            print(f"Skip {t}: {e}")
            continue

    if not temp_list_for_scaling:
        print("❌ Tidak ada data valid.")
        return

    # Fit Scaler ke SEMUA data gabungan
    combined_data = np.vstack(temp_list_for_scaling)
    scaler.fit(combined_data)
    
    # Simpan Scaler Baru (Menimpa yang lama)
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaler diperbarui & disimpan.")

    # 3. Buat Sequence Data (X dan y)
    for vals in processed_data_list:
        scaled_vals = scaler.transform(vals)
        X, y = create_sequences(scaled_vals, SEQ_LEN)
        all_X.append(X)
        all_y.append(y)
    
    final_X = np.vstack(all_X)
    final_y = np.concatenate(all_y)
    
    print(f"🧠 Training pada {len(final_X)} sampel data...")

    # 4. Arsitektur Model LSTM (Robust)
    model = Sequential([
        # Gaussian Noise untuk simulasi "Jitter" / Gangguan pasar
        GaussianNoise(0.02, input_shape=(SEQ_LEN, 8)), 
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Output Probabilitas 0-1
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(final_X, final_y, epochs=25, batch_size=32, verbose=1)
    
    # Simpan Model Baru (Menimpa yang lama)
    model.save('model_lstm.keras')
    print("✅ Model baru disimpan: model_lstm.keras")

if __name__ == "__main__":
    train()
