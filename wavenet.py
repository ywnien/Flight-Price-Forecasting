import argparse

import joblib
from keras.utils import set_random_seed
from keras.backend import clear_session

from util.models import early_stopping, tensor_board, BriefOuput
from util.models.cnn import wavenet, cond_wavenet
from util.models.helper import last_timestep_categorical_mae, last_timestep_mae
from util.snapshot import SplitData


def main(debug):
    causal = joblib.load("WindowData_causal.gz")
    splits = SplitData(causal)

    filters = 32
    n_blocks = 3
    layers_per_block = 4

    model_comment = "wavenet"
    sub_comment = "cond_relu"

    best_fare_mae = []
    print(f"{model_comment}_{sub_comment}")
    for n, fold in enumerate(splits):
        n += 1 # start from 1
        print(f"fold_{n} start")
        # initialization
        clear_session()
        set_random_seed(42)
        
        # model = wavenet(filters, n_blocks, layers_per_block)
        model = cond_wavenet(filters, n_blocks, layers_per_block, "relu")

        if debug:
            callbacks = [BriefOuput()]
        else:
            callbacks = [
                early_stopping(30),
                BriefOuput()
            ]

        history = model.fit(
            **fold,
            batch_size=32,
            epochs=3000,
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )
        best_fare_mae.append(f"{min(history.history['val_fare_mae']):.4f}")
        print(f"fold_{n} end")
        break

    print(f"{model_comment}_{sub_comment}")
    print("best val fare mae:", end=" ")
    print("\t".join(best_fare_mae))


def weight_test(debug, weights: list[float]):
    causal = joblib.load("data/WindowData_causal.gz")
    splits = SplitData(causal)
    fold_0 = splits[0]

    filters = 32
    n_blocks = 3
    layers_per_block = 4

    model_comment = "wavenet"
    sub_comment = f"cond_relu_{weights[0]}_{weights[1]}"

    print(f"{model_comment}_{sub_comment}")

    clear_session()
    set_random_seed(42)
        
    model = cond_wavenet(filters, n_blocks, layers_per_block, "relu")
    model.compile(
        optimizer="adam",
        loss={"fare": "mse", "seat": "sparse_categorical_crossentropy"},
        loss_weights={"fare": weights[0], "seat": weights[1]},
        metrics={"fare": last_timestep_mae, "seat": last_timestep_categorical_mae}
    )

    if debug:
        callbacks = [BriefOuput()]
    else:
        callbacks = [
            tensor_board(model_comment, f"{sub_comment}"),
            early_stopping(30),
            BriefOuput()
        ]

    history = model.fit(
        **fold_0,
        batch_size=32,
        epochs=3000,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    print(f"{model_comment}_{sub_comment}")
    print(f"best val fare mae: {min(history.history['val_fare_mae']):.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running causal_conv_lstm model")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    main(debug=args.debug)
