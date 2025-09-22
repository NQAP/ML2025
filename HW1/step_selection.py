import numpy as np
import pandas as pd

def ols_fit(X, y):
    """計算 OLS 迴歸係數與 SSE"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    y_pred = X @ beta
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    return beta, sse

def aic(sse, n, k):
    """計算 AIC"""
    return n * np.log(sse / n) + 2 * k

def stepwise_selection(X, y):
    """逐步回歸 (Forward + Backward, 使用 AIC 判斷)"""
    n, p = X.shape
    selected = []
    remaining = list(range(p))
    current_aic = np.inf

    while True:
        changed = False

        # ---------- Forward step ----------
        aic_candidates = []
        for feature in remaining:
            features_to_test = selected + [feature]
            X_test = np.c_[np.ones(n), X[:, features_to_test]]
            _, sse = ols_fit(X_test, y)
            k = len(features_to_test) + 1
            aic_val = aic(sse, n, k)
            aic_candidates.append((aic_val, feature))

        if aic_candidates:
            aic_candidates.sort()
            best_aic, best_feature = aic_candidates[0]

            if best_aic < current_aic:
                current_aic = best_aic
                selected.append(best_feature)
                remaining.remove(best_feature)
                changed = True
                print(f"Forward: 加入特徵 {best_feature}, AIC={best_aic:.2f}")

        # ---------- Backward step ----------
        if len(selected) > 1:
            aic_candidates = []
            for feature in selected:
                features_to_test = [f for f in selected if f != feature]
                X_test = np.c_[np.ones(n), X[:, features_to_test]]
                _, sse = ols_fit(X_test, y)
                k = len(features_to_test) + 1
                aic_val = aic(sse, n, k)
                aic_candidates.append((aic_val, feature))

            aic_candidates.sort()
            best_aic, worst_feature = aic_candidates[0]

            if best_aic < current_aic:
                current_aic = best_aic
                selected.remove(worst_feature)
                remaining.append(worst_feature)
                changed = True
                print(f"Backward: 移除特徵 {worst_feature}, AIC={best_aic:.2f}")

        if not changed:
            break

    return selected
