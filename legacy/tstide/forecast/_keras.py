from typing import Literal, Optional
import keras
import numpy as np
import pandas as pd

from ..dataframe import valid_folds
from ..transformer import TimeSeriesTransformer3D


def rolling_predict_keras(
    model: keras.Sequential,
    series: pd.DataFrame,
    n_steps: int,
    ts_trans: TimeSeriesTransformer3D,
):
    """
    - series: The dataframe includes futre n_steps dates.
    The number of rows must > lags + n_steps and exog variables must be filled.
    """
    arr_included = series.iloc[-(ts_trans.lags+n_steps):].copy().to_numpy()
    pred_array = np.zeros([n_steps+ts_trans.y_steps-1, len(ts_trans.series_names)])
    pred_count = np.zeros(n_steps+ts_trans.y_steps-1).reshape(-1, 1)

    series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
    time_step_left = n_steps
    for start_idx in range(n_steps-ts_trans.y_steps+1):
        # mid_idx = start_idx + ts_trans.lags
        end_idx = start_idx + ts_trans.lags + ts_trans.y_steps
        # TODO: Confirm, test the shape
        X = arr_included[start_idx:mid_idx].reshape(1, ts_trans.lags, len(series_indexes))
        y_pred = model.predict(X)
        assert len(y_pred.shape) == 2

        rolling_end = start_idx + ts_trans.y_steps
        pred_array[start_idx:rolling_end] = (
            pred_array[start_idx:rolling_end] * pred_count[start_idx:rolling_end] + y_pred
        )
        pred_count[start_idx:rolling_end] += 1
        pred_array[start_idx:rolling_end] /= pred_count[start_idx:rolling_end]

        arr_included[-n_steps:, series_indexes] = pred_array[:n_steps]

    return arr_included[-n_steps:, series_indexes]


def recursive_predict_keras(
    model: keras.Sequential,
    series: pd.DataFrame,
    n_steps: int,
    ts_trans: TimeSeriesTransformer3D,
):
    """
    - series: The dataframe includes futre n_steps dates.
    The number of rows must > lags + n_steps and exog variables must be filled.
    """
    arr_included = series.iloc[-(ts_trans.lags+n_steps):].copy().to_numpy()

    series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
    time_step_left = n_steps
    for start_idx in range(0, n_steps, ts_trans.y_steps):
        mid_idx = start_idx + ts_trans.lags
        end_idx = start_idx + ts_trans.lags + ts_trans.y_steps
        # TODO: Confirm, test the shape
        X = arr_included[start_idx:mid_idx].reshape(1, ts_trans.lags, len(series_indexes))
        y_pred = model.predict(X)
        assert len(y_pred.shape) == 2

        if ts_trans.y_steps > time_step_left:
            for i in range(time_step_left):
                arr_included.iloc[mid_idx+i, series_indexes] = y_pred[i]
            break

        arr_included[mid_idx:end_idx, series_indexes] = y_pred

        time_step_left -= ts_trans.y_steps
        start_idx += ts_trans.y_steps
        mid_idx += ts_trans.y_steps
        end_idx += ts_trans.y_steps

    return arr_included[-n_steps:, series_indexes]


def forecast_keras(
    reg: keras.Sequential,
    trailing_data: pd.DataFrame,
    ts_trans: TimeSeriesTransformer3D,
    n_steps: int = 7,
    method: Literal["recursive", "rolling"] = "recursive",
    decimals: Optional[tuple[int]] = (2, 0), # TODO: Change (This settings is only designed for our project)
):
    _methods = {"recursive": recursive_predict_keras, "rolling": rolling_predict_keras}
    pred_func = _methods[method]

    y_pred_arr = np.zeros(
        [trailing_data.index.get_level_values(0).unique().shape[0], n_steps, len(ts_trans.series_names)],
        dtype=np.float32
    )
    for row, _id in enumerate(trailing_data.index.get_level_values(0).unique()):
        pred_frame = pred_func(reg, trailing_data.loc[_id], n_steps, ts_trans)[-n_steps:]
        for index, (feature_name, _decimal) in enumerate(zip(ts_trans.series_names, decimals)):
            y_pred_arr[row, :, index] = np.around(pred_frame[feature_name].to_numpy().T, decimals=_decimal)

    return y_pred_arr


def backtesting_keras(
    model: keras.Sequential,
    series: pd.DataFrame,
    ts_trans: TimeSeriesTransformer3D,
    n_folds: int = 5,
    n_steps: int = 7,
    method: Literal["recursive", "rolling"] = "recursive",
    return_cache: bool = False,
    _dataset_cache: Optional[dict] = {},
):
    if return_cache:
        return _dataset_cache

    lags_steps = (ts_trans.lags, ts_trans.y_steps)
    y_pred_list = []
    for fold_n, valid_fold in enumerate(
        valid_folds(
            series,
            pd.to_datetime(series.index.get_level_values(1)),
            n_folds,
            n_steps
        )
    ):
        if (ts_trans.lags, ts_trans.y_steps) not in _dataset_cache:
            _dataset_cache[lags_steps] = {
                "X_train": {},
                "y_train": {},
                "trailing_data": {}
            }
        if (
            # caching
            fold_n in _dataset_cache["X_train"] and
            fold_n in _dataset_cache["y_train"] and
            fold_n in _dataset_cache["trailing_data"]
        ):
            X_train = _dataset_cache[lags_steps]["X_train"][fold_n]
            y_train = _dataset_cache[lags_steps]["y_train"][fold_n]
            trailing_data = _dataset_cache[lags_steps]["trailing_data"][fold_n]
        else:
            X_train, y_train, trailing_data = ts_trans.train_test_split(valid_fold, n_steps)
            _dataset_cache[lags_steps]["X_train"][fold_n] = X_train
            _dataset_cache[lags_steps]["y_train"][fold_n] = y_train
            _dataset_cache[lags_steps]["trailing_data"][fold_n] = trailing_data

        model.fit(X_train, y_train)
        # TODO: Adding decimals as backtesting param
        y_pred = forecast_keras(model, trailing_data, ts_trans, n_steps, method)
        y_pred_list.append(y_pred)
        print(f"Backtesting fold {fold_n+1} finished")

    return y_pred_list
