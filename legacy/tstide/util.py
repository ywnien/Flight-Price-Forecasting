from sklearn.metrics import mean_absolute_error


def backtesting_fare_mae(true_list, pred_list):
    error_fare = 0
    for true, pred in zip(true_list, pred_list):
        error_fare += mean_absolute_error(true[..., 0], pred[..., 0])
    error_fare /= len(true_list)
    return error_fare

