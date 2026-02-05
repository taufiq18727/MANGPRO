import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import requests
import tensorflow as tf
from datetime import datetime

# --- CONFIG ---
BOT_TOKEN = "8599866641:AAHA7GxblUZ6jVedQ2UOniFWKqxBy6HMn3M"
CHAT_IDS = ["977432672", "864486458"]

target_tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "ARTO.JK",
    "ADRO.JK", "PTBA.JK", "PGAS.JK", "MEDC.JK", "ANTM.JK", "MDKA.JK", 
    "TLKM.JK", "ISAT.JK", "EXCL.JK", "GOTO.JK", "BUKA.JK", "ASII.JK",
    "ICBP.JK", "INDF.JK", "UNVR.JK", "AMRT.JK", "BSDE.JK", "CTRA.JK", 
    "PANI.JK", "BRMS.JK", "BUMI.JK", "PSAB.JK", "DOID.JK", "ACES.JK",
    "SMRA.JK", "PWON.JK", "MAPA.JK"
]

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for uid in CHAT_IDS:
        try:
            requests.post(url, json={"chat_id": uid, "text": msg, "parse_mode": "Markdown"})
        except: pass

def predict_daily():
    print("🔍 Memulai Prediksi Harian...")

    try:
        # Load model tanpa compile agar tidak error masalah Focal Loss custom object
        model = tf.keras.models.load_model('model_lstm.keras', compile=False)
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        print(f"❌ Error Loading: {e}")
        return

    report_lines = []
    
    # Download data agak banyak untuk perhitungan indikator
    data = yf.download(target_tickers, period="6mo", auto_adjust=True, group_by='ticker', threads=True)
    
    for t in target_tickers:
        try:
            # --- FIX INDENTATION BLOCK ---
            df = data[t].copy()
            if len(df) < 100: continue
            
            # --- FEATURE ENGINEERING (WAJIB SAMA DENGAN TRAIN) ---
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            
            df['Vol_Change'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            df['MA50'] = df['Close'].rolling(50).mean()
            df['Dist_MA50'] = (df['Close'] - df['MA50']) / df['MA50']
            
            df = df.dropna()
            
            # Ambil 60 data terakhir
            cols = ['Log_Ret', 'RSI', 'MACD', 'Vol_Change', 'Dist_MA50']
            last_seq = df[cols].values[-60:]
            
            if len(last_seq) < 60: continue
            
            # Scaling
            input_scaled = scaler.transform(last_seq)
            input_data = np.expand_dims(input_scaled, axis=0)
            
            # Prediksi
            prob = model.predict(input_data, verbose=0)[0][0]
            price = df.iloc[-1]['Close']
            
            # --- LOGIC OUTPUT ---
            # Karena pakai Focal Loss, output biasanya terpolarisasi (sangat kecil atau sangat besar)
            # Kita filter yang > 50%
            if prob > 0.50: 
                # Icon khusus untuk probabilitas tinggi
                icon = "💎" if prob > 0.75 else "⚡"
                line = f"{icon} {t.replace('.JK',''):<5}| {int(price):<6}| {int(prob*100)}%"
                report_lines.append(line)
                
        except Exception as e:
            # print(f"Error {t}: {e}") # Debugging only
            continue

    # Kirim Laporan
    date_now = datetime.now().strftime("%d-%m-%Y")
    msg = f"🦅 *EAGLE EYE PRO V3*\n🗓 {date_now}\n"
    msg += f"_System: Focal Loss + Log Return_\n"
    msg += "—" * 15 + "\n"
    msg += "`Sts  Saham  Harga   Prob`\n```\n"
    
    report_lines.sort(key=lambda x: int(x.split('|')[2].replace('%','')), reverse=True)
    
    if report_lines:
        msg += "\n".join(report_lines)
    else:
        msg += "Market tidak jelas.\nAI memilih *Wait & See* (No Signal)."
        
    msg += "\n```\n💡 *Disclaimer On*"
    send_telegram(msg)
    print("✅ Selesai.")

if __name__ == "__main__":
    predict_daily()
