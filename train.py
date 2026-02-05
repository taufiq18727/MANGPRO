import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
import ta # Technical Analysis Library

# --- KONFIGURASI ---
TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", 
    "MDKA.JK", "GOTO.JK", "ADRO.JK", "UNVR.JK", "ICBP.JK", 
    "AMRT.JK", "KLBF.JK", "INCO.JK", "BRIS.JK"
]
SEQ_LEN = 60       # Lookback window
PREDICT_DAYS = 3   # Prediksi profit 3 hari ke depan
TARGET_PCT = 0.02  # Target: Kenaikan minimal 2% (Filter Noise)

def add_technical_indicators(df):
    df = df.copy()
    
    # 1. Log Returns (Agar data stasioner - KUNCI KEBERHASILAN AI)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. RSI (Momentum)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14) / 100.0 # Normalize 0-1
    
    # 3. MACD (Trend) - Normalize dengan membagi harga
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Diff'] = macd.macd_diff() / df['Close'] 
    
    # 4. Bollinger Bands %B (Posisi harga relatif thd Volatilitas)
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Pband'] = indicator_bb.bollinger_pband()
    
    # 5. Volume Change (Relatif)
    df['Vol_Change'] = np.log(df['Volume'] / df['Volume'].shift(1).replace(0, 1))

    # Drop NaN akibat calculation
    df.dropna(inplace=True)
    
    # Pilih fitur final
    # Log_Ret menangkap pergerakan harga, sisanya adalah konteks
    features = ['Log_Ret', 'RSI', 'MACD_Diff', 'BB_Pband', 'Vol_Change']
    return df[features]

def create_sequences(data, seq_length, raw_close):
    xs, ys = [], []
    for i in range(len(data) - seq_length - PREDICT_DAYS):
        x = data[i:(i + seq_length)]
        
        # Logic Target: Apakah harga Close 3 hari ke depan > Close hari ini + 2%?
        current_price = raw_close[i + seq_length - 1]
        future_max_price = np.max(raw_close[i + seq_length : i + seq_length + PREDICT_DAYS])
        
        # Label 1 jika potensi profit > 2%, else 0
        if future_max_price > current_price * (1 + TARGET_PCT):
            y = 1
        else:
            y = 0
            
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    print("🚀 Memulai Training Advanced AI...")
    
    all_X, all_y = [], []
    
    # Download data panjang (3 tahun) untuk menangkap berbagai kondisi market
    raw_data = yf.download(TICKERS, period="3y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    
    # Scaler menggunakan RobustScaler (Tahan terhadap outlier/spike harga ekstrim)
    scaler = RobustScaler()
    
    # --- PHASE 1: COLLECT DATA FOR SCALING ---
    training_data_cache = []
    
    for t in TICKERS:
        try:
            df = raw_data[t].copy()
            if len(df) < 200: continue
            
            # Simpan Raw Close untuk perhitungan target nanti
            raw_close = df['Close'].values
            
            # Feature Engineering
            df_features = add_technical_indicators(df)
            
            # Align raw_close dengan df_features (karena ada dropna)
            raw_close = raw_close[-len(df_features):]
            
            vals = df_features.values
            training_data_cache.append((vals, raw_close))
            
        except Exception as e:
            print(f"Skipped {t}: {e}")
            
    if not training_data_cache:
        print("❌ Data kosong.")
        return

    # Fit Scaler pada semua data gabungan agar model paham skala global
    combined_vals = np.vstack([x[0] for x in training_data_cache])
    scaler.fit(combined_vals)
    joblib.dump(scaler, 'scaler_advanced.pkl')
    print("✅ Scaler Robust Tersimpan.")
    
    # --- PHASE 2: CREATE SEQUENCES ---
    for vals, raw_close in training_data_cache:
        scaled_vals = scaler.transform(vals)
        X, y = create_sequences(scaled_vals, SEQ_LEN, raw_close)
        all_X.append(X)
        all_y.append(y)
        
    final_X = np.vstack(all_X)
    final_y = np.concatenate(all_y)
    
    # Balancing Class (Opsional, tapi bagus agar model tidak bias ke 'Sell')
    # Di saham bearish, label 1 akan jarang. 
    
    print(f"🧠 Training Data Shape: {final_X.shape}")
    
    # --- MODEL ARCHITECTURE (BIDIRECTIONAL LSTM) ---
    model = Sequential([
        # Bidirectional memungkinkan AI melihat pola maju & mundur
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LEN, 5)),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(), # Stabilize learning
        
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early Stopping agar tidak Overfitting (Hafalan)
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(final_X, final_y, epochs=30, batch_size=64, callbacks=[es], verbose=1)
    
    model.save('model_advanced.keras')
    print("✅ Model Canggih Tersimpan: model_advanced.keras")

if __name__ == "__main__":
    train_model()
