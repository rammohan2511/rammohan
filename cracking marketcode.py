import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import yfinance as yf
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates

# --- Validate ticker ---
def is_valid_ticker(ticker):
    try:
        data = yf.download(ticker, period="5d")
        return not data.empty
    except:
        return False

# --- Get next trading day (excluding weekends & NSE holidays) ---
def get_next_trading_day(current_date):
    nse_holidays = [
        "2025-01-26", "2025-03-14", "2025-04-18",
        "2025-08-15", "2025-10-02", "2025-11-04"
    ]
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in nse_holidays:
        next_day += timedelta(days=1)
    return next_day

# --- Download stock data ---
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        return df[['Close']] if not df.empty else None
    except:
        return None

# --- Preprocess for LSTM ---
def preprocess(data):
    close_prices = data['Close'].values.astype(float).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    x, y = [], []
    for i in range(60, len(scaled)):
        x.append(scaled[i - 60:i, 0])
        y.append(scaled[i, 0])

    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

# --- Build LSTM model ---
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Predict and plot ---
def predict_and_plot(ticker):
    for widget in frame.winfo_children():
        widget.destroy()

    if not is_valid_ticker(ticker):
        messagebox.showerror("Invalid Ticker", f"'{ticker}' is not a valid stock symbol.")
        return

    data = get_stock_data(ticker)
    if data is None or data.empty:
        messagebox.showerror("Data Error", f"Failed to retrieve data for '{ticker}'.")
        return

    data = data[data.index >= '2000-01-01']
    if len(data) < 80:
        messagebox.showerror("Data Error", f"Not enough data for '{ticker}'. Minimum 80 days required.")
        return

    try:
        x, y, scaler = preprocess(data)
        model = build_model((x.shape[1], 1))
        model.fit(x, y, epochs=5, batch_size=32, verbose=0)

        last_60 = data['Close'].values[-60:].reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60)
        last_60_scaled = np.reshape(last_60_scaled, (1, 60, 1))
        predicted_scaled = model.predict(last_60_scaled)
        predicted_price = float(scaler.inverse_transform(predicted_scaled).flatten()[0])

        yesterday_price = float(data['Close'].values[-1])
        last_date = data.index[-1]
        next_trading_day = get_next_trading_day(last_date)
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#f0f2f5')
    ax.set_facecolor('#ffffff')

    actual_prices = data['Close'].values.astype(float)
    plot_dates = data.index.to_pydatetime().tolist()
    extended_dates = plot_dates + [next_trading_day]
    extended_prices = np.append(actual_prices, predicted_price)

    # Gradient line
    points = np.array([np.arange(len(plot_dates)), actual_prices]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(plot_dates))
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
    lc.set_array(np.linspace(0, 1, len(plot_dates)))
    ax.add_collection(lc)

    # Final trend with prediction
    ax.plot(extended_dates, extended_prices, color='#00b7eb', linewidth=2, label='Price Trend', alpha=0.7)
    ax.plot(next_trading_day, predicted_price, 'o', color='#ff4500', markersize=10,
            label=f"Predicted: ₹{predicted_price:.2f}")

    ax.fill_between([plot_dates[-1], next_trading_day],
                    [yesterday_price, predicted_price - predicted_price * 0.05],
                    [yesterday_price, predicted_price + predicted_price * 0.05],
                    color='#ff4500', alpha=0.2, label='Prediction Range (±5%)')

    ax.set_title(f"{ticker.upper()} - Stock Price and Prediction for {next_trading_day.strftime('%Y-%m-%d')}",
                 fontsize=14, fontweight='bold', color='#333333')
    ax.set_xlabel("Date", fontsize=12, color='#333333')
    ax.set_ylabel("Price (₹)", fontsize=12, color='#333333')
    ax.legend(facecolor='#ffffff', edgecolor='#333333')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim([datetime(2000, 1, 1), extended_dates[-1]])
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    cbar = fig.colorbar(cm.ScalarMappable(cmap='viridis', norm=norm), ax=ax, label='Time Progression')
    cbar.set_label('Time Progression', fontsize=10, color='#333333')

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Hover function
    def on_hover(event):
        if event.inaxes == ax and event.xdata:
            numeric_dates = mdates.date2num(extended_dates)
            distance = np.abs(numeric_dates - event.xdata)
            nearest_index = distance.argmin()
            nearest_date = extended_dates[nearest_index]
            nearest_price = extended_prices[nearest_index]
            for txt in ax.texts:
                txt.remove()
            ax.text(nearest_date, nearest_price,
                    f'{nearest_date.strftime("%Y-%m-%d")}\n₹{nearest_price:.2f}',
                    color='black', fontsize=9, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
            canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", on_hover)

    # Display labels
    tk.Label(frame, text=f"Last Closing Price ({last_date.strftime('%Y-%m-%d')}): ₹{yesterday_price:.2f}",
             font=("Arial", 14), fg='#333333', bg='#f0f2f5').pack(pady=5)
    tk.Label(frame, text=f"Predicted Price ({next_trading_day.strftime('%Y-%m-%d')}): ₹{predicted_price:.2f}",
             font=("Arial", 14), fg='#ff4500', bg='#f0f2f5').pack(pady=5)

# --- GUI Setup ---
root = tk.Tk()
root.title("Cracking the Market Code: AI Stock Predictor")
root.geometry("780x720")
root.configure(bg='#f0f2f5')

tk.Label(root, text="Enter NSE Stock Ticker (e.g., RELIANCE.NS):",
         font=("Arial", 14), bg='#f0f2f5', fg='#333333').pack(pady=10)
ticker_entry = tk.Entry(root, font=("Arial", 14), width=25)
ticker_entry.pack(pady=5)

frame = tk.Frame(root, bg='#f0f2f5')
frame.pack()

tk.Button(root, text="Predict Price", font=("Arial", 14), bg='#ff4500', fg='#ffffff',
          command=lambda: predict_and_plot(ticker_entry.get().upper())).pack(pady=20)

root.mainloop()
