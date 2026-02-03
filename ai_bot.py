import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from datetime import datetime

# --- KONFIGURASI ---
BOT_TOKEN = "8599866641:AAHA7GxblUZ6jVedQ2UOniFWKqxBy6HMn3M"
CHAT_IDS = ["977432672", "864486458"]

tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", # Banking
    "ADRO.JK", "PTBA.JK", "PGAS.JK", "RAJA.JK", "MEDC.JK", # Energy
    "ANTM.JK", "MDKA.JK", "BRMS.JK", "TINS.JK", "PSAB.JK", # Minerals
    "TLKM.JK", "ISAT.JK", "GOTO.JK", "ASII.JK", "BUKA.JK", # Tech/Telco
    "PANI.JK", "BSDE.JK", "CTRA.JK", "SMRA.JK"             # Property
]

# --- FUNGSI KIRIM TELEGRAM ---
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for user_id in CHAT_IDS:
        try:
            payload = {"chat_id": user_id, "text": message, "parse_mode": "Markdown"}
            requests.post(url, json=payload)
            time.sleep(0.5) 
        except Exception as e:
            print(f"Error sending to {user_id}: {e}")

# --- MACHINE LEARNING ENGINE ---
def add_indicators(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MA Ratios & Volatility
    df['SMA5_Ratio'] = df['Close'] / df['Close'].rolling(window=5).mean()
    df['SMA20_Ratio'] = df['Close'] / df['Close'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Return'] = df['Close'].pct_change()
    
    return df.dropna()

def predict_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", auto_adjust=True)
        if len(df) < 200: return None

        df = add_indicators(df)
        
        # Target: 1 jika besok naik, 0 jika tidak
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        features = ['RSI', 'SMA5_Ratio', 'SMA20_Ratio', 'Vol_Ratio', 'Return']
        
        # Training (Semua data kecuali 60 hari terakhir)
        train = df.iloc[:-60]
        test = df.iloc[-60:]
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=1)
        model.fit(train[features], train['Target'])
        
        # Evaluasi Akurasi (Precision)
        preds = model.predict(test[features])
        precision = precision_score(test['Target'], preds, zero_division=0)
        
        # Prediksi Besok
        last_day = df.iloc[[-1]][features]
        proba = model.predict_proba(last_day)[0]
        prob_up = proba[1] # Probabilitas Naik
        
        return {
            "price": df.iloc[-1]['Close'],
            "precision": precision,
            "prob_up": prob_up
        }
    except:
        return None

def run_analysis():
    print("Memulai Analisa AI...")
    
    report_lines = []
    
    for t in tickers:
        res = predict_stock(t)
        if not res: continue
        
        symbol = t.replace(".JK", "")
        # Filter: Hanya tampilkan jika Probabilitas > 50%
        if res['prob_up'] > 0.50:
            emoji = "🟢" if res['prob_up'] >= 0.60 else "⚪"
            line = f"{emoji} {symbol:<5} | {int(res['price']):<6} | {int(res['prob_up']*100)}%"
            report_lines.append(line)
            
    # Menyusun Pesan Telegram
    date_now = datetime.now().strftime("%d-%m-%Y")
    
    # Header
    msg = f"🤖 *AI PREDICTION REPORT*\n🗓 {date_now}\n"
    msg += f"_Metode: Random Forest (ML)_\n"
    msg += "—" * 15 + "\n"
    msg += "`Sts  Saham  Harga   Peluang`\n" # Header Tabel Monospace
    msg += "```\n" # Mulai blok kode agar tabel rapi
    
    # Isi Tabel (Diurutkan dari peluang tertinggi)
    # Sort based on probability descending
    report_lines.sort(key=lambda x: int(x.split('|')[2].replace('%','')), reverse=True)
    
    msg += "\n".join(report_lines)
    msg += "\n```" # Tutup blok kode
    
    # Legend & Disclaimer
    msg += "\n—" * 15
    msg += "\n🟢 = Potensi Kuat (>60%)"
    msg += "\n⚪ = Netral/Pantau (50-60%)"
    msg += "\n\n💡 *Disclaimer On*: AI hanya membaca pola data historis."

    print("Mengirim ke Telegram...")
    send_telegram(msg)
    print("Selesai.")

if __name__ == "__main__":
    run_analysis()
