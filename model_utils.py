# model_utils.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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
                epochs=5, batch_size=64, early_stop=True):

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
