import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import requests
import tensorflow as tf
from datetime import datetime

# --- KONFIGURASI ---
BOT_TOKEN = "8599866641:AAHA7GxblUZ6jVedQ2UOniFWKqxBy6HMn3M"
CHAT_IDS = ["977432672", "864486458"]

# Daftar saham yang mau di-scan setiap hari (Bisa beda sama training)
target_tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK",
    "ADRO.JK", "PTBA.JK", "PGAS.JK", "RAJA.JK", "MEDC.JK",
    "ANTM.JK", "MDKA.JK", "BRMS.JK", "TINS.JK", "PSAB.JK",
    "TLKM.JK", "ISAT.JK", "GOTO.JK", "ASII.JK", "BUKA.JK",
    "PANI.JK", "BSDE.JK", "CTRA.JK", "SMRA.JK" 
]

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for uid in CHAT_IDS:
        try:
            requests.post(url, json={"chat_id": uid, "text": message, "parse_mode": "Markdown"})
        except: pass

def predict_daily():
    print("🔍 Memulai Prediksi Harian...")
    
    # 1. Load Otak & Scaler Terbaru
    try:
        model = tf.keras.models.load_model('model_lstm.keras')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model & Scaler berhasil dimuat.")
    except:
        print("❌ Model tidak ditemukan. Jalankan training dulu.")
        return

    report_lines = []
    
    # 2. Ambil data secukupnya (6 bulan terakhir)
    data = yf.download(target_tickers, period="6mo", auto_adjust=True, group_by='ticker', threads=True)
    
    for t in target_tickers:
        try:
            df = data[t].copy()
            if len(df) < 100: continue
            
            # REPLIKASI FITUR (Harus sama persis dengan train.py)
            df['RSI'] = 100 - (100 / (1 + df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean()/(-df['Close'].diff().where(df['Close'].diff() < 0, 0)).rolling(14).mean()))
            df['Weekly_Trend'] = df['Close'] / df['Close'].shift(5)
            df['Vol_Change'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            df = df.dropna()
            
            # Ambil 60 data terakhir (SEQ_LEN)
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'Weekly_Trend', 'Vol_Change']
            last_seq = df[cols].values[-60:]
            
            if len(last_seq) < 60: continue
            
            # Scaling
            last_seq_scaled = scaler.transform(last_seq)
            
            # Reshape ke (1, 60, 8)
            input_data = np.expand_dims(last_seq_scaled, axis=0)
            
            # Prediksi
            prob = model.predict(input_data, verbose=0)[0][0]
            price = df.iloc[-1]['Close']
            
            # Hanya ambil yang probabilitasnya > 50%
            if prob > 0.50:
                emoji = "🟢" if prob >= 0.70 else "⚪"
                line = f"{emoji} {t.replace('.JK',''):<5}| {int(price):<6}| {int(prob*100)}%"
                report_lines.append(line)
                
        except Exception as e:
            continue

    # 3. Kirim Laporan
    date_now = datetime.now().strftime("%d-%m-%Y")
    msg = f"🧠 *AI PREDICTION UPDATE*\n🗓 {date_now}\n"
    msg += f"_Model: Retrained Weekly_\n"
    msg += "—" * 15 + "\n"
    msg += "`Sts  Saham  Harga   Prob`\n```\n"
    
    # Urutkan dari probabilitas tertinggi
    report_lines.sort(key=lambda x: int(x.split('|')[2].replace('%','')), reverse=True)
    
    if report_lines:
        msg += "\n".join(report_lines)
    else:
        msg += "Tidak ada sinyal kuat hari ini."
        
    msg += "\n```\n💡 *Disclaimer On*"
    
    send_telegram(msg)
    print("Selesai.")

if __name__ == "__main__":
    predict_daily()
