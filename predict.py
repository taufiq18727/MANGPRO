import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import requests
import ta
import tensorflow as tf
from datetime import datetime
import os

# --- SETUP TELEGRAM (Ambil dari Environment Variable GitHub) ---
BOT_TOKEN = os.getenv("BOT_TOKEN") 
CHAT_ID = os.getenv("CHAT_ID")

# Jika testing lokal, hardcode sementara (Jangan commit token asli ke public repo!)
# BOT_TOKEN = "TOKEN_ANDA"
# CHAT_ID = "ID_ANDA"

# List Saham Screening (Daftar Favorit Anda)
TARGET_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK",
    "ADRO.JK", "PTBA.JK", "PGAS.JK", "ANTM.JK", "MDKA.JK", "MBMA.JK",
    "TLKM.JK", "ISAT.JK", "JSMR.JK",
    "ICBP.JK", "MYOR.JK", "AMRT.JK", "MAPI.JK", "ACES.JK",
    "BSDE.JK", "CTRA.JK", "SMRA.JK", "PWON.JK",
    "GOTO.JK", "BUKA.JK", "EMTK.JK",
    "ASII.JK", "AUTO.JK", "DRMA.JK",
    "BRMS.JK", "BUMI.JK", "DOID.JK", "MAPA.JK", "ESSA.JK"
]

SEQ_LEN = 60

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram Token/Chat ID belum diset.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})

def add_technical_indicators(df):
    # Sama persis dengan fungsi training
    df = df.copy()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14) / 100.0
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Diff'] = macd.macd_diff() / df['Close']
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Pband'] = indicator_bb.bollinger_pband()
    df['Vol_Change'] = np.log(df['Volume'] / df['Volume'].shift(1).replace(0, 1))
    df.dropna(inplace=True)
    return df[['Log_Ret', 'RSI', 'MACD_Diff', 'BB_Pband', 'Vol_Change']]

def run_prediction():
    print("🔍 Menjalankan Screening Harian...")
    
    try:
        model = tf.keras.models.load_model('model_advanced.keras')
        scaler = joblib.load('scaler_advanced.pkl')
    except:
        print("❌ Model/Scaler tidak ditemukan. Jalankan train.py dulu.")
        return

    predictions = []
    
    # Download data 6 bulan terakhir
    data = yf.download(TARGET_TICKERS, period="6mo", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    
    for t in TARGET_TICKERS:
        try:
            df = data[t].copy()
            if len(df) < 100: continue
            
            # Ambil harga terakhir real untuk display
            last_price = df['Close'].iloc[-1]
            
            # Preprocessing
            df_features = add_technical_indicators(df)
            
            # Ambil sequence terakhir
            if len(df_features) < SEQ_LEN: continue
            last_seq = df_features.values[-SEQ_LEN:]
            
            # Scale & Reshape
            last_seq_scaled = scaler.transform(last_seq)
            input_data = np.expand_dims(last_seq_scaled, axis=0)
            
            # Predict
            prob = model.predict(input_data, verbose=0)[0][0]
            
            # Logika Sinyal (Hanya ambil yang confidence tinggi)
            # Threshold > 0.60 karena model baru lebih ketat
            if prob > 0.55: 
                signal_strength = "⚡" if prob > 0.75 else "🟢"
                predictions.append({
                    'ticker': t.replace('.JK',''),
                    'price': int(last_price),
                    'prob': prob,
                    'icon': signal_strength
                })
                
        except Exception as e:
            continue
            
    # Sort by Probability Tertinggi
    predictions.sort(key=lambda x: x['prob'], reverse=True)
    
    # Format Pesan Telegram
    date_now = datetime.now().strftime("%d-%m-%Y")
    msg = f"🦅 *EAGLE EYE AI PREDICTION*\n"
    msg += f"🗓 {date_now} | _Bi-LSTM Model_\n"
    msg += "—" * 15 + "\n"
    msg += "`Sts  Saham  Harga   Conf`\n"
    msg += "```\n"
    
    if predictions:
        for p in predictions[:15]: # Top 15 saja
            # Format: ⚡ BBCA  | 10200 | 85%
            msg += f"{p['icon']} {p['ticker']:<5}| {p['price']:<6}| {int(p['prob']*100)}%\n"
    else:
        msg += "Market mendung. Tidak ada sinyal kuat > 55%.\n"
        
    msg += "```\n💡 _Disclaimer: High Risk_"
    
    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    run_prediction()
