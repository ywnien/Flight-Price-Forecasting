from datetime import datetime

import joblib
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (LSTM, concatenate, Dense, Embedding, Input,
                          LayerNormalization, Reshape, Softmax,
                          TimeDistributed, Permute)
from keras.models import Model

from snapshot import LAGS, N_STEPS, WindowData, SplitData


model_comment = "Causal_LSTM"
embed_dim = 2
NUM_CLASSES = 11

causal = joblib.load("WindowData_causal.gz")
splits = SplitData(causal)
fold_0 = splits[0]


def last_timestep_mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true[:, -1], y_pred[:, -1])


encoded = Input([LAGS, 34], name="encoded")
fare_basis = Input(LAGS, name="fare_basis")
embedded = Embedding(input_dim=250, output_dim=embed_dim)(fare_basis)
embedded = Reshape((LAGS, embed_dim))(embedded)

cat_x = concatenate([encoded, embedded])
x = LSTM(32, return_sequences=True)(cat_x)
x = LSTM(32, return_sequences=True)(x)
x = LSTM(32, return_sequences=True)(x)
x = LSTM(32, return_sequences=True)(x)

reg = TimeDistributed(Dense(N_STEPS), name="reg")(x)
_clf_classes = concatenate([TimeDistributed(Dense(N_STEPS))(x) for _ in range(NUM_CLASSES)])
clf_classes = Permute([1, 3, 2])(Reshape([LAGS, NUM_CLASSES, N_STEPS])(_clf_classes))
clf = Softmax(name="clf")(clf_classes)

model = Model(inputs=[encoded, fare_basis], outputs=[reg, clf])
model.compile(
    optimizer="adam",
    loss={"reg": "mse", "clf": "sparse_categorical_crossentropy"},
    metrics={"reg": last_timestep_mae},
)
model.summary()


# TIMESTAMP = datetime.now().strftime("%m%d_%H%M")
# tensor_board = TensorBoard(
#     log_dir=f"log/{model_comment}/{TIMESTAMP}/", histogram_freq=1
# )
# early_stopping = EarlyStopping(
#     patience=30, monitor="val_reg_mae", mode="min", restore_best_weights=True
# )

history = model.fit(
    **fold_0,
    epochs=3000,
    batch_size=32,
    # callbacks=[early_stopping, tensor_board],
    shuffle=False,
    verbose=1,
)

# model.save(f"models/{model_comment}_{TIMESTAMP}_0.keras")
