import joblib
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (GRU, Concatenate, Dense, Embedding, Input,
                          LayerNormalization, RepeatVector, Reshape,
                          TimeDistributed, add)
from keras.models import Model

from snapshot import *

test_run_data = joblib.load("test_run_data.gz")


model_comment = "embed_dim_exp"

def make_model(embed_dim: int):
    clear_session()
    encoded = Input(shape=(LAGS, 34), name="encoded")
    fare_basis = Input(shape=(LAGS, 1), name="fare_basis")

    if embed_dim != 0:
        embedded = Embedding(input_dim=250, output_dim=embed_dim)(fare_basis)
        embedded = Reshape((LAGS, embed_dim))(embedded)
        concatenated_tensor = Concatenate(axis=-1)([encoded, embedded])
        concatenated_tensor = LayerNormalization()(concatenated_tensor)
    else:
        concatenated_tensor = LayerNormalization()(encoded)

    unit = concatenated_tensor.shape[-1]

    _lstm_output = LayerNormalization()(
        add([GRU(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(concatenated_tensor), concatenated_tensor])
    )
    _lstm_output = LayerNormalization()(
        add([GRU(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(_lstm_output), _lstm_output])
    )
    _lstm_output = LayerNormalization()(
        GRU(unit, kernel_regularizer="l2", dropout=0.1)(_lstm_output)
    )
    lstm_encode = RepeatVector(N_STEPS)(_lstm_output)

    _lstm_decode = LayerNormalization()(
        add([GRU(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(lstm_encode), lstm_encode])
    )
    lstm_output = LayerNormalization()(
        add([GRU(unit, kernel_regularizer="l2", dropout=0.1, return_sequences=True)(_lstm_decode), _lstm_decode])
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
        metrics={'reg': 'mae'}
    )

    model.summary()

    early_stopping = EarlyStopping(patience=30, monitor="val_reg_mae", mode="min", restore_best_weights=True)
    tensor_board = TensorBoard(log_dir=f"log/{model_comment}/lstm_{embed_dim}/", histogram_freq=1)

    model.fit(
        **test_run_data,
        epochs=3000,
        batch_size=32,
        callbacks=[early_stopping, tensor_board],
        shuffle=False,
        verbose=1,
    )

for num in (0, 2, 4, 8, 16, 32, 64):
    make_model(num)
