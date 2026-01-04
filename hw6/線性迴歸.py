import random
import math

# ----------------------------
# Data (你可以改成自己的資料)
# ----------------------------
x = [1.0, 2.0, 3.0]
y = [2.0, 3.0, 5.0]
n = len(x)

def predict(w, b, xi):
    return w * xi + b

def mse(w, b):
    return sum((predict(w, b, x[i]) - y[i])**2 for i in range(n)) / n

# ----------------------------
# 1) 改良法：解析解 / Normal Equation（1D 版本）
#    w = cov(x,y)/var(x)
#    b = y_mean - w*x_mean
# ----------------------------
def normal_equation_1d():
    x_mean = sum(x) / n
    y_mean = sum(y) / n

    var_x = sum((xi - x_mean)**2 for xi in x)
    if var_x == 0:
        raise ValueError("All x are identical -> cannot fit slope.")

    cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    w = cov_xy / var_x
    b = y_mean - w * x_mean
    return w, b

# ----------------------------
# 2) 爬山算法 Hill Climbing（小步試探四方向）
# ----------------------------
def hill_climbing(step=0.01, max_iters=50_000, patience=2_000):
    w, b = 0.0, 0.0
    best = mse(w, b)
    no_improve = 0

    for _ in range(max_iters):
        # 四個方向試探
        candidates = [
            (w + step, b),
            (w - step, b),
            (w, b + step),
            (w, b - step),
        ]
        cw, cb = min(candidates, key=lambda p: mse(p[0], p[1]))
        cand = mse(cw, cb)

        if cand < best:
            w, b = cw, cb
            best = cand
            no_improve = 0
        else:
            no_improve += 1

        # 走很久都沒變好：停止
        if no_improve >= patience:
            break

    return w, b, best

# ----------------------------
# 3) 貪婪法：Coordinate Descent（交替最佳化 w / b）
#    固定 b 求 w（用 1D 最小平方法的閉式解）
#    固定 w 求 b（b = mean(y - w*x)）
# ----------------------------
def coordinate_descent(max_iters=10_000, tol=1e-12):
    w, b = 0.0, 0.0
    prev = mse(w, b)

    x_mean = sum(x) / n
    var_x = sum((xi - x_mean)**2 for xi in x)
    if var_x == 0:
        raise ValueError("All x are identical -> cannot fit slope.")

    for _ in range(max_iters):
        # (A) 固定 b，更新 w
        # minimize sum((w*x_i + b - y_i)^2)
        # w = sum((x_i - x_mean)*(y_i - b - mean(y-b))) / var_x
        y_minus_b = [y[i] - b for i in range(n)]
        ymb_mean = sum(y_minus_b) / n
        cov = sum((x[i] - x_mean) * (y_minus_b[i] - ymb_mean) for i in range(n))
        w = cov / var_x

        # (B) 固定 w，更新 b：b = mean(y - w*x)
        b = sum(y[i] - w * x[i] for i in range(n)) / n

        cur = mse(w, b)
        if abs(prev - cur) < tol:
            break
        prev = cur

    return w, b, prev

# ----------------------------
# 4) 急劇下降法：Gradient Descent
#    dJ/dw = (2/n) * sum((w*x_i + b - y_i)*x_i)
#    dJ/db = (2/n) * sum((w*x_i + b - y_i))
# ----------------------------
def gradient_descent(lr=0.05, max_iters=100_000, tol=1e-12):
    w, b = 0.0, 0.0
    prev = mse(w, b)

    for _ in range(max_iters):
        # 計算梯度
        err = [predict(w, b, x[i]) - y[i] for i in range(n)]
        dw = (2.0 / n) * sum(err[i] * x[i] for i in range(n))
        db = (2.0 / n) * sum(err)

        # 更新
        w -= lr * dw
        b -= lr * db

        cur = mse(w, b)
        if abs(prev - cur) < tol:
            break
        prev = cur

    return w, b, prev

# ----------------------------
# Run all
# ----------------------------
if __name__ == "__main__":
    w1, b1 = normal_equation_1d()
    print("[1] Normal Equation")
    print("   w =", w1, "b =", b1, "mse =", mse(w1, b1))

    w2, b2, e2 = hill_climbing(step=0.01)
    print("\n[2] Hill Climbing")
    print("   w =", w2, "b =", b2, "mse =", e2)

    w3, b3, e3 = coordinate_descent()
    print("\n[3] Coordinate Descent (Greedy)")
    print("   w =", w3, "b =", b3, "mse =", e3)

    w4, b4, e4 = gradient_descent(lr=0.05)
    print("\n[4] Gradient Descent")
    print("   w =", w4, "b =", b4, "mse =", e4)
