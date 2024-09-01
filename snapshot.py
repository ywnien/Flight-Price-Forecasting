from copy import copy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from tqdm import tqdm

sns.set_theme()

LAGS = 14
N_STEPS = 7
VALIDATION_FOLDS = 5


def convert_timestamps(df: pd.DataFrame):
    df["searchDate"] = pd.to_datetime(df["searchDate"])
    df["segmentsDepartureTimeEpochSeconds"] = pd.to_datetime(
        df["segmentsDepartureTimeEpochSeconds"], unit="s"
    )
    return df


def filter_and_sort(df: pd.DataFrame):
    # Tickets with duration smaller than `LAGS + N_STEPS` were filtered
    group = df.groupby("legId")["searchDate"]
    duration = group.max() - group.min() + pd.Timedelta(days=1)
    duration = duration[duration >= pd.Timedelta(days=LAGS + N_STEPS)]
    leg_valid = duration.index
    df = df[df["legId"].isin(leg_valid)]
    # sort searchDate by their minimum date and then maximum date
    sorted_idx = np.lexsort((group.max()[leg_valid], group.min()[leg_valid]))
    sorted_legs = leg_valid[sorted_idx]

    return df.set_index("legId").loc[sorted_legs].reset_index()


def leg_slice_generator(df: pd.DataFrame):
    pos = 0
    for span in df["legId"].value_counts(sort=False):
        yield slice(pos, pos + span)
        pos += span


def ffill_bfill(arr: np.ndarray):
    """
    Peforming foward filling and backward filling. The index of elements would be
    created, and the indexes of nan values were set to 0.Then, doing accumulate
    maximum to find the mamimum available indexes at current position. Finally,
    get the filled array by the resulting indexes.
    """
    mask = np.isnan(arr)
    # return array of row indexes and the index of nan is set to 0
    idx = np.where(~mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    # calcuating accumulate maximum in reverse order is back filling
    np.maximum.accumulate(idx[::-1], axis=0, out=idx)
    idx = idx[::-1]
    return arr[idx, np.arange(arr.shape[1])]


def impute_null_data(df: pd.DataFrame):
    arr = df[["segmentsEquipmentDescription", "totalTravelDistance"]].to_numpy()
    enc = OrdinalEncoder()
    arr = enc.fit_transform(arr)

    for leg_slice in leg_slice_generator(df):
        subarr = arr[leg_slice]
        if np.isnan(subarr).any() and np.isnan(subarr).all() != True:
            subarr[:] = ffill_bfill(subarr)

    arr = enc.inverse_transform(arr)
    df["segmentsEquipmentDescription"] = arr[:, 0]
    df["totalTravelDistance"] = arr[:, 1].astype(np.float32)

    return df


def fill_distance(df: pd.DataFrame):
    distance_airport_map = {"ONT": 1897, "LAX": 1943}
    df["totalTravelDistance"].fillna(
        df["segmentsArrivalAirportCode"].map(distance_airport_map), inplace=True
    )
    return df


def fill_equipment(df: pd.DataFrame):
    null = df["segmentsEquipmentDescription"].isna()
    spirit = df["segmentsAirlineName"] == "Spirit Airlines"
    delta = df["segmentsAirlineName"] == "Delta"
    df.loc[
        null & spirit, "segmentsEquipmentDescription"
    ] = "AIRBUS INDUSTRIE A320 SHARKLETS"
    df.loc[null & delta, "segmentsEquipmentDescription"] = "Airbus A321"

    return df


def impute_lost_days(df: pd.DataFrame):
    df_arr = df.to_numpy()
    search_date_index = df.columns.get_loc("searchDate")
    search_date = df_arr[:, search_date_index].astype("datetime64[D]")

    # losing data or changing legId caused the searchDate difference not equal 1
    date_diff = np.diff(search_date) // np.timedelta64(1, "D")
    # excluding the part caused by changing legId
    leg_counts = df["legId"].value_counts(sort=False)
    leg_change_indexes = (leg_counts.to_numpy().cumsum() - 1)[:-1]
    date_diff[leg_change_indexes] = 1

    # calculate the corresponding indexes of existing data in the new array
    new_indexes = np.zeros(search_date.shape[0], dtype=int)
    date_diff.cumsum(out=new_indexes[1:])

    arr = np.zeros([new_indexes[-1] + 1, df.shape[1]], dtype=object)
    arr[new_indexes] = df_arr  # copy the existing data to the new array

    # impute part
    mark_arr = np.zeros(arr.shape[0], dtype=np.float32)
    DAY = np.timedelta64(1, "D")
    for i in np.nonzero(date_diff != 1)[0]:
        lost_days = date_diff[i] - 1
        start = new_indexes[i] + 1
        stop = start + lost_days
        arr[start:stop] = df_arr[i]  # forward fill
        mark_arr[start:stop] = 1  # mark the imputed data
        ref_day = np.datetime64(df_arr[i, search_date_index], "D")
        arr[start:stop, search_date_index] = np.arange(
            ref_day + DAY, ref_day + DAY * date_diff[i], DAY
        )

    array_dict = {
        key: arr[:, i].astype(type_) for i, (key, type_) in enumerate(df.dtypes.items())
    }
    array_dict.update({"imputed": mark_arr})
    return array_dict


@cache
def month_start(year, month):
    return pd.Timestamp(year=year, month=month, day=1)


def cyclic_encode(
    timestamps: ArrayLike, period: Literal["day", "week", "month", "year"]
):
    """
    Encoding features which can be converted by `pd.to_datetime`.
    The returning features consist of sine and cosine waves with period determined
    by the parameter `period`.

    Parameters
    ----------
    timestamps : ArrayLike object of timestamps
        Could be converted by `pd.to_datetime`
    period : "day", "week", "month" or "year"
        The period in sine and cosine wave

    Returns
    -------
    tuple(x_sin, x_cos)
    """
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)

    if period == "day":
        offset = (timestamps - timestamps.dt.normalize()).dt.total_seconds()
        _period = 86400
    elif period == "week":
        # seconds of days passed this week + hour, minute and seconds
        offset = timestamps.dt.day_of_week * 86400
        offset += (timestamps - timestamps.dt.normalize()).dt.total_seconds()
        _period = 86400 * 7
    elif period == "month":
        # offset to the the beginning of the month
        offset = timestamps.apply(
            lambda x: x - month_start(x.year, x.month)
        ).dt.total_seconds()
        _period = timestamps.dt.days_in_month * 86400  # 86400 seconds in a day
    elif period == "year":
        offset = timestamps.dt.day_of_year * 86400
        offset += (timestamps - timestamps.dt.normalize()).dt.total_seconds()
        _period = 86400 * 365
    else:
        raise ValueError("The parameter period only support day, week, month and year")

    basis = 2 * np.pi * offset / _period
    basis = basis.to_numpy(np.float32)
    return np.sin(basis), np.cos(basis)


class EncodeData:
    target_names = ("totalFare", "seatsRemaining")
    encoding_spec = {
        "Ordinal": ["fareBasisCode"],
        "OneHot": [
            "segmentsAirlineName",
            "segmentsArrivalAirportCode",
            "segmentsEquipmentDescription",
        ],
    }
    encoders = {}

    def __init__(self, df: pd.DataFrame):
        if df is None:
            return
        self.data = self.clean_impute_df(df)
        self.metadata = {
            "legId": self.data.pop("legId"),
            "searchDate": self.data["searchDate"],
            "imputed": self.data.pop("imputed"),
        }
        self.columns, self.x, self.y = self.encode()

    @classmethod
    def read_csv(cls, filepath: str, **kwargs):
        return cls(pd.read_csv(filepath, **kwargs))

    @staticmethod
    def clean_impute_df(df: pd.DataFrame) -> dict[str, np.ndarray]:
        selected_features = [
            "searchDate",
            "segmentsDepartureTimeEpochSeconds",
            "legId",
            "fareBasisCode",
            "segmentsArrivalAirportCode",
            "segmentsAirlineName",
            "segmentsEquipmentDescription",
            "totalFare",
            "seatsRemaining",
            "isBasicEconomy",
            "totalTravelDistance",
            "segmentsDurationInSeconds",
        ]
        return (
            df.loc[:, selected_features]
            .pipe(convert_timestamps)
            .pipe(filter_and_sort)
            .pipe(impute_null_data)
            .pipe(fill_distance)
            .pipe(fill_equipment)
            .pipe(impute_lost_days)
        )

    def fit_encoders(self):
        search_date = self.data["searchDate"]
        last_day = search_date.max() - pd.Timedelta(days=N_STEPS * VALIDATION_FOLDS)
        train_mask = search_date <= last_day

        for feature in self.encoding_spec["Ordinal"]:
            self.encoders[feature] = OrdinalEncoder(
                dtype=np.float32,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.encoders[feature].fit(self.data[feature][train_mask].reshape(-1, 1))
        for feature in self.encoding_spec["OneHot"]:
            self.encoders[feature] = OneHotEncoder(
                dtype=np.float32,
                handle_unknown="infrequent_if_exist",
                min_frequency=0.01,
                sparse_output=False,
                feature_name_combiner=self.feature_name_combiner(feature),
            )
            self.encoders[feature].fit(self.data[feature][train_mask].reshape(-1, 1))

    @staticmethod
    def feature_name_combiner(feature):
        def combiner(_, category):
            return f"{feature}_{category}"

        return combiner

    def encode(self):
        if not self.encoders:
            self.fit_encoders()

        # Ordinal encode
        embed_data = {}
        for feature in self.encoding_spec["Ordinal"]:
            arr = self.data.pop(feature)
            enc = self.encoders[feature]
            # +1 to make unkown value become 0. Embedding layer only accept values >= 0
            embed_data[feature] = enc.transform(arr.reshape(-1, 1)) + 1
        # One Hot encode
        for feature in self.encoding_spec["OneHot"]:
            arr = self.data.pop(feature)
            enc = self.encoders[feature]
            arr = enc.transform(arr.reshape(-1, 1))
            self.data.update(
                {name: arr[:, i] for i, name in enumerate(enc.get_feature_names_out())}
            )
        # Cyclic encode
        period_args = {
            "searchDate": ["week", "month", "year"],
            "segmentsDepartureTimeEpochSeconds": ["day", "week", "month", "year"],
        }
        for feature, period_list in period_args.items():
            arr = self.data.pop(feature)
            for period in period_list:
                waves = cyclic_encode(arr, period=period)
                descr = f"{feature}_{period}"
                self.data.update({f"{descr}_sin": waves[0], f"{descr}_cos": waves[1]})

        # return (columns, x, y)
        return (
            tuple(tuple(data.keys()) for data in (self.data, embed_data)),
            tuple(
                np.column_stack(tuple(data.values())).astype(np.float32)
                for data in (self.data, embed_data)
            ),
            np.column_stack(tuple(self.data[name] for name in self.target_names)),
        )


@dataclass
class WindowData:
    x: tuple[np.ndarray]
    y: np.ndarray
    imputed_x: np.ndarray
    imputed_y: np.ndarray
    date: np.ndarray

    @classmethod
    def sliding_window(cls, encoded: EncodeData):
        frag = {
            "x": tuple([] for _ in encoded.x),
            "y": [],
            "imputed_x": [],
            "imputed_y": [],
            "date": [],
        }
        search_date = encoded.metadata["searchDate"].reshape(-1, 1)
        imputed = encoded.metadata["imputed"].reshape(-1, 1)

        for leg_slice in cls._leg_slice_generator(encoded.metadata["legId"]):
            # append windowed samples
            for frag_x, arr in zip(frag["x"], encoded.x):
                frag_x.append(cls._window_x(arr[leg_slice]))
            frag["y"].append(cls._window_y(encoded.y[leg_slice], causal=True))
            frag["imputed_x"].append(cls._window_x(imputed[leg_slice]))
            frag["imputed_y"].append(cls._window_y(imputed[leg_slice], causal=True))
            # use last date of samples to split train, validation and test set
            frag["date"].append(search_date[leg_slice][LAGS+N_STEPS-1:])

        # concatenate windowed samples
        causal = cls(**frag)
        causal._vstack_samples()
        # sort data by the order of date
        ind = np.argsort(causal.date)
        causal._apply_index(ind)
        # window is for non causal model whose output is future timesteps
        window = copy(causal)
        window.y = causal.y[:, -1]  # output at last timestep is future timesteps
        window.imputed_y = causal.imputed_y[:, -1]
        # convert imputed counts to imputed rate (percentage)
        causal.imputed_x = window.imputed_x = cls._imputed_rate(causal.imputed_x)
        causal.imputed_y = cls._imputed_rate(causal.imputed_y)
        window.imputed_y = cls._imputed_rate(window.imputed_y)

        return window, causal

    def _vstack_samples(self):
        for field_name in self.__dataclass_fields__:
            attr = getattr(self, field_name)
            if isinstance(attr, list) and attr:
                attr = np.vstack(attr).squeeze() # squeeze to remove redundant dim
            elif isinstance(attr, tuple): # special case for x
                attr = tuple(np.vstack(arr) for arr in attr)
            else:
                continue
            setattr(self, field_name, attr)

    @staticmethod
    def _leg_slice_generator(leg_id_array: np.ndarray):
        leg_counts = pd.Series(leg_id_array).value_counts(sort=False)
        pos = 0
        for span in leg_counts:
            yield slice(pos, pos + span)
            pos += span

    @staticmethod
    def _window_x(array: np.ndarray):
        return np.lib.stride_tricks.sliding_window_view(
            array[:-N_STEPS], window_shape=LAGS, axis=0
        ).swapaxes(1, 2)

    @staticmethod
    def _window_y(array: np.ndarray, causal: bool = False):
        if causal:
            return np.lib.stride_tricks.sliding_window_view(
                array[1:], window_shape=(N_STEPS, LAGS), axis=(0, 0)
            ).swapaxes(1, 3)
        else:
            return np.lib.stride_tricks.sliding_window_view(
                array[LAGS:], window_shape=N_STEPS, axis=0
            ).swapaxes(1, 2)

    @staticmethod
    def _imputed_rate(imputed: np.ndarray):
        axis_to_sum = tuple(range(1, imputed.ndim))
        return imputed.sum(axis=axis_to_sum) / np.prod(imputed.shape[1:])

    def _apply_index(self, ind: np.ndarray):
        for field_name in self.__dataclass_fields__:
            attr = getattr(self, field_name)
            if isinstance(attr, np.ndarray):
                attr = attr[ind]
            elif isinstance(attr, tuple):
                attr = tuple(arr[ind] for arr in attr)
            else:
                continue
            setattr(self, field_name, attr)


class SplitData:
    def __init__(self, window: WindowData, n_folds: int = VALIDATION_FOLDS) -> None:
        self.window = window
        self.n_folds = n_folds
        self._normalization_params = [None for _ in range(n_folds)]

    # def train_valid_split(self, n_folds: int = VALIDATION_FOLDS):
    #     x, y = self.window.x, self.window.y
    #     train_boundary = self.window.date.max() - np.timedelta64(N_STEPS * n_folds, "D")
    #     for index in range(n_folds):
    #         valid_boundary = train_boundary + np.timedelta64(N_STEPS, "D")
    #         train_mask = self.window.date <= train_boundary
    #         valid_mask = (self.window.date <= valid_boundary) ^ train_mask
    #         kwargs = {
    #             "x": [arr[train_mask] for arr in x],
    #             "y": [arr[train_mask] for arr in np.rollaxis(y, axis=-1)],
    #             "validation_data": (
    #                 [arr[valid_mask] for arr in x],
    #                 [arr[valid_mask] for arr in np.rollaxis(y, axis=-1)],
    #             ),
    #             "sample_weight": self.sample_weight[train_mask],
    #         }
    #         yield self._normalize_split(kwargs, index)
    #         # yield kwargs
    #         train_boundary = valid_boundary

    def train_valid_split(self, n_folds: int = VALIDATION_FOLDS):
        for i in range(n_folds):
            yield self[i]

    def __getitem__(self, index: int):
        if index >= self.n_folds:
            raise IndexError(f"index out of range. Index should <= {self.n_folds}")

        train_boundary = self.window.date.max() - np.timedelta64(
            N_STEPS * (self.n_folds - index), "D"
        )
        valid_boundary = train_boundary + np.timedelta64(N_STEPS, "D")
        train_mask = self.window.date <= train_boundary
        valid_mask = (self.window.date <= valid_boundary) ^ train_mask
        kwargs = {
            "x": [arr[train_mask] for arr in self.window.x],
            "y": [arr[train_mask] for arr in np.rollaxis(self.window.y, axis=-1)],
            "validation_data": (
                [arr[valid_mask] for arr in self.window.x],
                [arr[valid_mask] for arr in np.rollaxis(self.window.y, axis=-1)],
            ),
            "sample_weight": self.sample_weight[train_mask],
        }
        return self._normalize_split(kwargs, index)

    @property
    def sample_weight(self):
        return (1 - self.window.imputed_x) * (1 - self.window.imputed_y)

    def _normalize_split(self, split_dict, index):
        params = {}  # parameters are mean and std
        # features to embed do not need normalization
        x_train, x_valid = split_dict["x"], split_dict["validation_data"][0]
        norm_train, norm_valid, mean, std = self.normalize(x_train[0], x_valid[0])
        x_train[0], x_valid[0] = norm_train, norm_valid
        params["x"] = (mean, std)

        # only normalize the regression target: totalFare
        y_train, y_valid = split_dict["y"], split_dict["validation_data"][1]
        norm_train, norm_valid, mean, std = self.normalize(y_train[0], y_valid[0])
        y_train[0], y_valid[0] = norm_train, norm_valid
        params["y"] = (mean, std)
        self._normalization_params[index] = params  # save params in object
        return split_dict

    @staticmethod
    def normalize(train: np.ndarray, valid: np.ndarray):
        # if the array only contains 1 feature, the shape would be (samples, timesteps)
        ndim = max(train.ndim, 3)  # ensure calling mean/std like 3d array
        axes = tuple(range(ndim - 1))
        mean = train.mean(axis=axes)
        std = train.std(axis=axes)
        return (train - mean) / std, (valid - mean) / std, mean, std

    def denormalize_fare(self, array: np.ndarray, index: int = -1):
        mean, std = self._normalization_params[index]["y"]
        return array * std + mean

    # TODO: developing, need to transform to methods
    # def save_fold(fold, cache_path):
    #     var_nums = {"x": 2, "y": 2}
    #     val_pos = {"x": 0, "y": 1}
    #     kwargs = {}
    #     for var, num in var_nums.items():
    #         for i in range(num):
    #             kwargs[f"{var}_{i}"] = fold[var][i]
    #             kwargs[f"validation_data_{var}_{i}"] = fold[f"validation_data"][val_pos[var]][i]
    #     kwargs["sample_weight"] = fold["sample_weight"]
    #     np.savez_compressed(cache_path, **kwargs)

    # def load_fold(cache_path):
    #     var_nums = {"x": 2, "y": 2}
    #     fold = {}
    #     with np.load(cache_path) as fp:
    #         fold["x"] = [fp[f"x_{i}"] for i in range(var_nums["x"])]
    #         fold["y"] = [fp[f"y_{i}"] for i in range(var_nums["y"])]
    #         fold["validation_data"] = [
    #             [fp[f"validation_data_{var}_{i}"] for i in range(var_nums[var])]
    #             for var in var_nums.keys()
    #         ]
    #         fold["sample_weight"] = fp["sample_weight"]
    #     return fold


def collapse_2d(x: np.ndarray):
    last_x = x[:, -1]
    # [0, 1] are indexes of totalFare and seatsRemaining
    targets_x = x[:, :-1, [0, 1]].reshape(x.shape[0], -1)
    return np.hstack((targets_x, last_x))


def sk_model_kwargs(fold: dict, regression_only: bool = False):
    x = collapse_2d(np.concatenate(fold["x"], axis=-1))
    x_valid = collapse_2d(np.concatenate(fold["validation_data"][0], axis=-1))
    y = fold["y"]
    if regression_only:
        y = np.concatenate(y, axis=-1)
        return {
            "fit": ((x, y), {"sample_weight": fold["sample_weight"]}),
            "predict": {"X": x_valid},
        }
    return (
        {
            "fit": ((x, y[0]), {"sample_weight": fold["sample_weight"]}),
            "predict": {"X": x_valid},
        },
        {
            "fit": ((x, y[1]), {"sample_weight": fold["sample_weight"]}),
            "predict": {"X": x_valid},
        },
    )


def _predict_reg_only(fold: dict, reg) -> np.ndarray:
    sk_kwargs = sk_model_kwargs(fold, regression_only=True)
    return reg.fit(*sk_kwargs["fit"][0], **sk_kwargs["fit"][1]).predict(
        **sk_kwargs["predict"]
    )


def _predict_reg_clf(fold: dict, reg, clf) -> tuple[np.ndarray]:
    sk_kwargs = sk_model_kwargs(fold)
    # cloning a new, not referenced model to make sure it is freed after prediciton
    return tuple(
        clone(model)
        .fit(*kwargs["fit"][0], **kwargs["fit"][1])
        .predict(**kwargs["predict"])
        for kwargs, model in zip(sk_kwargs, (reg, clf))
    )


def sk_model_predict(window: WindowData, name: str, reg=None, clf=None):
    splits = SplitData(window)
    fare_list = []
    seat_list = []
    for i, fold in tqdm(
        enumerate(splits.train_valid_split()), total=VALIDATION_FOLDS, desc=name
    ):
        if clf is None:
            y_pred = _predict_reg_only(fold, reg)  # np.ndarray
            fare_list.append(splits.denormalize_fare(y_pred[:, :N_STEPS], i))
            seat_list.append(y_pred[:, N_STEPS:].round())
        else:
            y_pred = _predict_reg_clf(fold, reg, clf)  # tuple[np.ndarray]
            fare_list.append(splits.denormalize_fare(y_pred[0], i))
            seat_list.append(y_pred[1])
    np.savez_compressed(
        f"models/{name}.npz", fare=np.vstack(fare_list), seat=np.vstack(seat_list)
    )
    print(f"y_pred saved at 'models/{name}.npz'")


def load_y_pred(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        y_preds = dict(f)
    return y_preds


def _melt_relplot(fare_mae: pd.DataFrame, seat_mae: pd.DataFrame):
    fare_mae["day"] = np.arange(N_STEPS) + 1
    fare_mae["target"] = "totalFare"
    seat_mae["day"] = np.arange(N_STEPS) + 1
    seat_mae["target"] = "seatsRemaining"
    mae = pd.concat([fare_mae, seat_mae])
    melt = pd.melt(mae, ["day", "target"], var_name="model", value_name="MAE")

    by = ["target", "model"]
    style = style_order = None
    if melt["model"].str.contains("_reg_only").any():
        style = "reg_only"
        by.append(style)
        style_order = ["Yes", "No"]
        melt["reg_only"] = np.where(
            melt["model"].str.contains("_reg_only"), "No", "Yes"
        )
        melt["model"] = melt["model"].str.replace("_reg_only", "")

    sns.relplot(
        melt,
        x="day",
        y="MAE",
        hue="model",
        style=style,
        kind="line",
        style_order=style_order,
        col="target",
        facet_kws={"sharey": False},
    )
    display(melt.drop(columns="day").groupby(by).agg(["mean", "max", "min"]))


def plot_mae(name_and_path: dict):
    y_true = load_y_pred("models/y_true.npz")
    fare_mae = {}
    seat_mae = {}
    for name, path in name_and_path.items():
        y_pred = load_y_pred(path)
        y_pred["seat"] = y_pred["seat"].round()  # TODO: make sure rounded while saving
        for mae, target in zip((fare_mae, seat_mae), ("fare", "seat")):
            mae[name] = mean_absolute_error(
                y_true[target], y_pred[target], multioutput="raw_values"
            )
    _melt_relplot(pd.DataFrame(fare_mae), pd.DataFrame(seat_mae))


def error_dist(pred_path: str, days_avg: bool = True):
    y_true = load_y_pred("models/y_true.npz")
    y_pred = load_y_pred(pred_path)
    target_spec = {"fare": "totalFare", "seat": "seatsRemaining"}
    hue = None
    palette = None
    if days_avg:
        errors = {
            name: (y_pred[key] - y_true[key]).mean(axis=-1)
            for key, name in target_spec.items()
        }
        errors = pd.DataFrame(errors).melt(value_name="error", var_name="target")
    else:
        hue = "days"
        palette = "crest"
        errors = pd.concat(
            pd.DataFrame(y_pred[key] - y_true[key], columns=np.arange(7) + 1)
            .melt(value_name="error", var_name=hue)
            .assign(target=name)
            for key, name in {"fare": "totalFare", "seat": "seatsRemaining"}.items()
        )
    sns.displot(
        errors,
        x="error",
        col="target",
        kind="kde",
        hue=hue,
        palette=palette,
        facet_kws={"sharex": False, "sharey": False},
    )
