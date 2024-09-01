from datetime import datetime

import joblib
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (LSTM, Concatenate, Dense, Embedding, Input,
                          LayerNormalization, RepeatVector, Reshape,
                          TimeDistributed, add)
from keras.models import Model

from snapshot import *

LAGS = 14
N_STEPS = 7
test_run_data = joblib.load("test_run_data.gz")

window = joblib.load("WindowData_window.gz")
splits = SplitData(window)
fold_1 = next(splits.train_valid_split())



# embed_dim = 8
embed_dim = 4

encoded = Input(shape=(LAGS, 34), name="encoded")
fare_basis = Input(shape=(LAGS, 1), name="fare_basis")

embedded = Embedding(input_dim=250, output_dim=embed_dim)(fare_basis)
embedded = Reshape((LAGS, embed_dim))(embedded)
concatenated_tensor = Concatenate(axis=-1)([encoded, embedded])


model_comment = "LSTM_encoder_decoder"

unit = concatenated_tensor.shape[-1]

_lstm_output = LayerNormalization()(
    add([LSTM(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(concatenated_tensor), concatenated_tensor])
)
_lstm_output = LayerNormalization()(
    add([LSTM(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(_lstm_output), _lstm_output])
)
_lstm_output = LayerNormalization()(
    LSTM(unit, kernel_regularizer="l2", dropout=0.1)(_lstm_output)
)
lstm_encode = RepeatVector(N_STEPS)(_lstm_output)

_lstm_decode = LayerNormalization()(
    add([LSTM(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(lstm_encode), lstm_encode])
)
lstm_output = LayerNormalization()(
    add([LSTM(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(_lstm_decode), _lstm_decode])
)

# Regression output
lstm_output = TimeDistributed(
    Dense(16, activation="relu")
)(lstm_output)
reg_output = TimeDistributed(
    Dense(1, activation='linear'), name='reg'
)(lstm_output)

# Classification output
clf_output = TimeDistributed(
    Dense(11, activation='softmax'), name="clf"
)(lstm_output)

model = Model(inputs=[encoded, fare_basis], outputs=[reg_output, clf_output])

model.compile(
    optimizer='adam',
    loss={'reg': 'mse', 'clf': 'sparse_categorical_crossentropy'},
    metrics={'reg': 'mae', 'clf': 'accuracy'}
)

model.summary()


TIMESTAMP = datetime.now().strftime("%m%d_%H%M")
tensor_board = TensorBoard(log_dir=f"log/{model_comment}_{TIMESTAMP}/", histogram_freq=1)
early_stopping = EarlyStopping(patience=30, monitor="val_reg_mae", mode="min", restore_best_weights=True)

splits_ = SplitData(window)
last_epoch = 0
for i, fold in enumerate(splits_.train_valid_split()):
    print(last_epoch)
    history = model.fit(
        **fold,
        epochs=300,
        batch_size=32,
        initial_epoch=last_epoch,
        callbacks=[early_stopping, tensor_board],
        shuffle=False,
        verbose=0,
    )
    last_epoch += len(history.epoch)

    model.save(f"models/{model_comment}_{TIMESTAMP}_{i}.keras")
