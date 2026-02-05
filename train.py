import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler

# --- CONFIG ---
tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", 
    "ASII.JK", "ANTM.JK", "MDKA.JK", "GOTO.JK",
    "ADRO.JK", "PGAS.JK", "UNVR.JK", "ICBP.JK",
    "BRMS.JK", "PANI.JK", "AMMN.JK", "PSAB.JK", 
    "MEDC.JK", "AKRA.JK", "ISAT.JK", "ACES.JK"
]

SEQ_LEN = 60 

# --- CUSTOM LOSS: FOCAL LOSS ---
# Senjata rahasia mengatasi probabilitas 50%
def focal_loss(gamma=2., alpha=4.):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) -tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

def add_features(df):
    df = df.copy()
    
    # 1. Log Return (PENTING: Mengubah Harga Rupiah jadi Persentase)
    # Ini membuat GOTO dan BBCA setara di mata AI
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Trend Filter)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # 4. Volatility Ratio
    df['Vol_Change'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # 5. Distance from MA50 (Trend Strength)
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Dist_MA50'] = (df['Close'] - df['MA50']) / df['MA50']
    
    return df.dropna()

def train():
    print("🚀 Memulai Training Advanced (Focal Loss)...")
    
    # Download data agak panjang agar MA50 terbentuk
    raw_data = yf.download(tickers, period="2y", auto_adjust=True, group_by='ticker', threads=True)
    
    # RobustScaler lebih tahan terhadap lonjakan harga tiba-tiba
    scaler = RobustScaler()
    
    all_sequences = []
    all_targets = []
    
    print("⚙️ Processing Data & Creating Sequences...")
    for t in tickers:
        try:
            df = raw_data[t].copy()
            if len(df) < 200: continue
            
            # Simpan harga Close asli untuk penentuan Target nanti
            close_prices = df['Close'].values
            
            df = add_features(df)
            
            # Fitur yang masuk ke AI (Semua dalam bentuk rasio/persen, bukan Rupiah)
            cols = ['Log_Ret', 'RSI', 'MACD', 'Vol_Change', 'Dist_MA50']
            feature_data = df[cols].values
            
            # Kita perlu align data karena add_features membuang baris awal (NaN)
            # Hitung offset data yang hilang
            start_idx = len(close_prices) - len(feature_data)
            aligned_close = close_prices[start_idx:]
            
            # Scaling per ticker (Penting agar distribusi data seragam)
            scaled_data = scaler.fit_transform(feature_data)
            
            for i in range(len(scaled_data) - SEQ_LEN - 1):
                x = scaled_data[i : i + SEQ_LEN]
                
                # --- LOGIC TARGET (Strict Mode) ---
                # Harga saat ini (H) vs Harga Besok (H+1)
                price_today = aligned_close[i + SEQ_LEN - 1]
                price_tomorrow = aligned_close[i + SEQ_LEN]
                
                # Target: HARUS NAIK > 1% (Minimal)
                # Jika cuma naik 0.5% atau turun, dianggap 0.
                if price_tomorrow > price_today * 1.01:
                    y = 1
                else:
                    y = 0
                
                all_sequences.append(x)
                all_targets.append(y)
                
        except Exception as e:
            print(f"Skip {t}: {e}")
            continue

    if not all_sequences:
        print("❌ Data kosong/gagal diproses.")
        return

    X = np.array(all_sequences)
    y = np.array(all_targets)
    
    print(f"📊 Total Dataset: {len(X)} samples")
    print(f"🔥 Jumlah Sinyal BUY (>1% gain): {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
    print("   (Jika persentase kecil, itu BAGUS. AI belajar mencari jarum di tumpukan jerami)")

    # Simpan Dummy Scaler untuk Predict nanti
    # Kita fit ulang ke seluruh data agar punya parameter global
    scaler.fit(np.vstack(all_sequences).reshape(-1, 5)) 
    joblib.dump(scaler, 'scaler.pkl')

    # --- ARSITEKTUR MODEL (Deep LSTM) ---
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, 5)),
        Dropout(0.4),
        BatchNormalization(), # Stabilizer
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile dengan Focal Loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss=focal_loss(), 
                  metrics=['accuracy'])
    
    # Early Stopping agar tidak overtrain
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(X, y, epochs=40, batch_size=64, verbose=1, callbacks=[es])
    
    model.save('model_lstm.keras')
    print("✅ Model Advanced Disimpan.")

if __name__ == "__main__":
    train()
