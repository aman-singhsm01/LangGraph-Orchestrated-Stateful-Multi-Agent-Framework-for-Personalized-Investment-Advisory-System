import pandas as pd
import numpy as np
import yfinance as yf
import pywt
from tensorflow.keras import backend as K
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, GRU, Bidirectional
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def fetch_and_preprocess_data(ticker, period="5y"):
    print(f"\n--- Fetching 5-Year Data for {ticker} ---")
    df = yf.download(ticker, period=period, threads=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.copy(deep=True).astype('float64')
        
    print(f"\nDataset Shape: {df.shape}")
    print("--- First 5 Rows of the Dataset ---")
    print(df.head(5)) 
    print("-" * 35)
    
    df['RSI'] = RSIIndicator(df['Close'].copy()).rsi()
    macd = MACD(df['Close'].copy())
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    
    coeffs = pywt.wavedec(df['Close'].to_numpy(copy=True), 'db4', level=2)
    coeffs[1:] = [pywt.threshold(i, value=0.5, mode='soft') for i in coeffs[1:]]
    df['Denoised_Close'] = pywt.waverec(coeffs, 'db4')[:len(df)]
    
    return df

def evaluate_models(df):
    print("\n--- Running Model Training & Evaluation (All Algorithms) ---")
    
    L = len(df)
    train_size = int(L * 0.8)
    test_len = L - train_size
    
    # --- MEMORY UNLOCK FIX 2 ---
    # Force a deep copy of the numpy arrays so scikit-learn can scale them
    test_actual = df['Denoised_Close'].values[-test_len:].copy()
    
    predictions = {}
    
    # 1. ARIMA Model
    train_arima = df['Denoised_Close'].values[:train_size].copy()
    history = list(train_arima)
    arima_preds = []
    for t in range(test_len):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        arima_preds.append(yhat)
        history.append(test_actual[t])
    predictions['ARIMA'] = np.array(arima_preds)

    # --- Deep Learning Data Prep (Prices + Indicators) ---
    dl_features = df[['Denoised_Close', 'RSI', 'MACD', 'MACD_Signal']].values.copy()
    dl_target = df[['Denoised_Close']].values.copy()
    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(dl_features)
    scaled_target = target_scaler.fit_transform(dl_target)
    
    X_dl, y_dl = [], []
    look_back = 10
    for i in range(len(scaled_features) - look_back):
        X_dl.append(scaled_features[i:(i + look_back), :])
        y_dl.append(scaled_target[i + look_back, 0])
        
    X_dl = np.array(X_dl)
    y_dl = np.array(y_dl)
    
    X_train_dl, y_train_dl = X_dl[:-test_len].copy(), y_dl[:-test_len].copy()
    X_test_dl = X_dl[-test_len:].copy()

    # 2. LSTM
    K.clear_session()
    lstm_model = Sequential([Input(shape=(10, 4)), LSTM(50), Dense(1)])
    lstm_model.compile(loss='mse', optimizer='adam')
    lstm_model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=16, verbose=0)
    predictions['LSTM'] = target_scaler.inverse_transform(lstm_model.predict(X_test_dl, verbose=0)).flatten()
    # 3. GRU
    K.clear_session()
    bilstm_model = Sequential([Input(shape=(10, 4)), Bidirectional(LSTM(50)), Dense(1)])
    bilstm_model.compile(loss='mse', optimizer='adam')
    bilstm_model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=16, verbose=0)
    predictions['BiLSTM'] = target_scaler.inverse_transform(bilstm_model.predict(X_test_dl, verbose=0)).flatten()
    # 4. BiLSTM
    bilstm_model = Sequential([Input(shape=(10, 4)), Bidirectional(LSTM(50)), Dense(1)])
    bilstm_model.compile(loss='mse', optimizer='adam')
    bilstm_model.fit(X_train_dl, y_train_dl, epochs=5, batch_size=16, verbose=0)
    predictions['BiLSTM'] = target_scaler.inverse_transform(bilstm_model.predict(X_test_dl, verbose=0)).flatten()

    # --- Standard ML Data Prep (Prices + Indicators) ---
    X_ml_list = []
    y_ml_list = []
    
    for i in range(len(df) - look_back):
        history_prices = df['Denoised_Close'].values[i:(i + look_back)].copy()
        indicators = df[['RSI', 'MACD', 'MACD_Signal']].values[i + look_back - 1].copy()
        
        combined_features = np.concatenate((history_prices, indicators))
        X_ml_list.append(combined_features)
        y_ml_list.append(df['Denoised_Close'].values[i + look_back])
        
    X_ml = np.array(X_ml_list)
    y_ml = np.array(y_ml_list)
    
    X_train_ml, y_train_ml = X_ml[:-test_len].copy(), y_ml[:-test_len].copy()
    X_test_ml = X_ml[-test_len:].copy()

    # 5. Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train_ml, y_train_ml)
    predictions['Random Forest'] = rf_model.predict(X_test_ml)
    
    # 6. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    xgb_model.fit(X_train_ml, y_train_ml)
    predictions['XGBoost'] = xgb_model.predict(X_test_ml)
    
    # 7. AdaBoost
    ada_model = AdaBoostRegressor(n_estimators=50, random_state=42)
    ada_model.fit(X_train_ml, y_train_ml)
    predictions['AdaBoost'] = ada_model.predict(X_test_ml)

    # 8. Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train_ml, y_train_ml)
    predictions['Decision Tree'] = dt_model.predict(X_test_ml)

    # 9. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_ml, y_train_ml)
    predictions['Linear Regression'] = lr_model.predict(X_test_ml)

    # --- Calculate Metrics & Find Top 3 ---
    metrics = {}
    model_mse = {}
    
    for name, preds in predictions.items():
        mse = mean_squared_error(test_actual, preds)
        mae = mean_absolute_error(test_actual, preds)
        r2 = r2_score(test_actual, preds)
        metrics[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        model_mse[name] = mse

    top_3_names = sorted(model_mse, key=model_mse.get)[:3]
    
    # 10. Dynamic Top-3 Hybrid Model
    hybrid_preds = (predictions[top_3_names[0]] + predictions[top_3_names[1]] + predictions[top_3_names[2]]) / 3.0
    metrics['Top 3 Hybrid'] = {
        'MSE': mean_squared_error(test_actual, hybrid_preds),
        'MAE': mean_absolute_error(test_actual, hybrid_preds),
        'R2': r2_score(test_actual, hybrid_preds)
    }
    predictions['Top 3 Hybrid'] = hybrid_preds
    
    all_model_mse = {name: metrics[name]['MSE'] for name in metrics.keys()}
    best_model_name = min(all_model_mse, key=all_model_mse.get)
    best_pred = predictions[best_model_name][-1]

    # Print Table
    print("\n" + "="*80)
    print(f"{'Model Performance Metrics':^80}")
    print("="*80)
    print(f"{'Algorithm':<25} | {'MSE':<12} | {'MAE':<12} | {'R2 Score':<12}")
    print("-" * 80)
    for model_name, m in metrics.items():
        if model_name == best_model_name:
            print(f"{model_name + ' (BEST)':<25} | {m['MSE']:<12.4f} | {m['MAE']:<12.4f} | {m['R2']:<12.4f}")
        else:
            print(f"{model_name:<25} | {m['MSE']:<12.4f} | {m['MAE']:<12.4f} | {m['R2']:<12.4f}")
    print("="*80 + "\n")
    print(f"Top 3 Models forming Hybrid: {', '.join(top_3_names)}")
    print(f"Algorithm selected for final prediction: {best_model_name}")
    
    return best_pred, metrics, best_model_name