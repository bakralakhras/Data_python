from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


def build_model(input_shape,
                lstm_units=(32, 16),
                dropout_rate=0.2,
                dense_units=16):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units[0], return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units[1], return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, X_val, y_val,
                epochs=20, batch_size=64, early_stop=True):

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    callbacks = []
    if early_stop:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        ))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    preds = model.predict(X_val)
    return model, preds, history

def evaluate_model(y_test, predictions, scaler):
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(predictions)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae, y_true, y_pred

def plot_loss(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
def forecast_future(model, df, scaler, time_step, n_future, freq):
    """
    Predict n_future steps ahead with small random noise for demo visuals.
    """
    last_sequence = df['Price'].values[-time_step:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, time_step, 1)

    future_predictions = []
    current_input = last_sequence_scaled

    for _ in range(n_future):
        next_pred_scaled = model.predict(current_input, verbose=0)[0][0]

        noise = np.random.uniform(-0.002, 0.002) 
        next_pred_scaled = next_pred_scaled * (1 + noise)

        future_predictions.append(next_pred_scaled)

        next_input = np.append(current_input.flatten()[1:], [next_pred_scaled])
        current_input = next_input.reshape(1, time_step, 1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_future+1, freq=freq)[1:]

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
    future_df.set_index('Date', inplace=True)
    return future_df


