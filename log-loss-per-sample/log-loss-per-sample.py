import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    yc = np.clip(y_pred, eps, 1 - eps) # clip the prediction values
    ll = -1 * (y_true * np.log(yc) + np.subtract(1, y_true) * np.log(np.subtract(1, yc)))
    return list(ll)