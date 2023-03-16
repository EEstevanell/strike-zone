from copy import copy
from typing import Tuple
import numpy as np
from numpy.random import default_rng
import math
rng = default_rng()

class Geo:
    def fit(self, X, y):
        pass

    def predict(self, X: np.ndarray):
        pass
    
    def distance(self, point):
        pass

    def plot(self, xs, ys, title):
        pass

class Line(Geo):
    """
    y = mx + n
    """

    def __init__(self, slope=None, y_intercept=None):
        self.slope=slope
        self.y_intersect=y_intercept

    def __repr__(self) -> str:
        return f"Line (y = ({self.slope})x + ({self.y_intersect}))" if self.slope else "Line (Not fitted)"
    
    def distance(self, point):
        x,y = point
        a = self.slope
        b = -1
        c = self.y_intersect
        return abs(a*x + b*y + c)/math.sqrt(a**2 + b**2)

    def fit(self, X, y):
        self.slope = (y[1] - y[0]) / (X[1] - X[0])
        self.y_intersect = (X[1]*y[0] - X[0] *y[1]) / (X[1] - X[0])
        return self
    
    def predict(self, X: np.ndarray):
        return np.array([(self.slope * x) + self.y_intersect for x in X])
    
    def plot(self, xs, ys, title):
        import matplotlib.pyplot as plt

        plt.scatter(xs, ys)
        
        x_max = max(xs)
        x_min = min(xs)
        
        line = np.linspace(x_min, x_max, num=200)
        predicted = self.predict(line)

        plt.plot(line, predicted)

        # Adding the title
        plt.title(title)

        # Adding the labels
        plt.ylabel("Y")
        plt.xlabel("X")

        plt.show()

class Parabola(Geo):
    "y = ax**2 + bx + c"
    
    def __init__(self, a=None, b=None, c=None):
        self.a=a
        self.b=b
        self.c=c

    def __repr__(self) -> str:
        return f"Parabola (y = ({self.a})x**2 + ({self.b})x + ({self.c}))" if self.a else "Parabola (Not fitted)"
    
    def distance(self, point):
        x1,y1 = point
        def p_dist(x, y):
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5

        def dist_func(x):
            return p_dist(x, self.predict([x])[0])

        from scipy.optimize import minimize_scalar
        res = minimize_scalar(dist_func)
        return res.fun

    def fit(self, X, y):
        x1, x2, x3 = X
        y1, y2, y3 = y

        A : np.matrix = np.matrixlib.asmatrix([
                [x1**2, x1, 1],
                [x2**2, x2, 1],
                [x3**2, x3, 1]
            ])
        
        C : np.matrix = np.matrixlib.asmatrix([
            [y1],
            [y2],
            [y3]
        ])

        # resolving:
        #     [a]   
        # A * |b| = C
        #     [v]   

        Ainv = np.linalg.inv(A)

        results = np.linalg.multi_dot([Ainv, C])
        self.a = float(results[0][0])
        self.b = float(results[1][0])
        self.c = float(results[2][0])

        return self
    
    def predict(self, X: np.ndarray):
        return np.array([
            self.a*(x**2) + self.b*x + self.c for x in X
            ])
    
class RANSAC:
    def __init__(self, n=10, k=100, t=10, d=2, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.inlier_X = None
        self.inlier_y = None

    def fit(self, X, y):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0])

            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

            loss = (
                self.loss(
                    maybe_model, 
                    ids[self.n :], 
                    y[ids][self.n :], 
                    maybe_model.predict(X[ids][self.n :]))
                )
            
            thresholded = [x for x in loss if x <= self.t]
            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size >= self.d:
                this_error = self.metric(maybe_model, X, y, maybe_model.predict(X), self.t)

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model
                    
                    point_indxs = in_line_indxs(maybe_model, X, y, maybe_model.predict(X), self.t)
                    self.inlier_X = [X[i] for i in point_indxs]
                    self.inlier_y =  [y[i] for i in point_indxs]

        return self

    def predict(self, X):
        return self.best_fit.predict(X)


# General Loss and Error

def square_error_loss(model, x, y_true, y_pred):
    return (y_true - y_pred) ** 2

def mean_square_error(model, x_true, y_true, y_pred):
    return np.sum(square_error_loss(model, x_true, y_true, y_pred)) / y_true.shape[0]


# Geo Loss and Error

def min_distance_loss(line: Geo, x_true, y_true, y_pred):
    result = []
    for i in range(len(x_true)):
        x = x_true[i]
        y = y_true[i]
        result.append(line.distance((x, y)))
    return result

def mean_contain_error(line: Geo, x_true, y_true, y_pred, threshold):
    losses = min_distance_loss(line, x_true, y_true, y_pred)
    return np.sum(losses) / y_true.shape[0]

def in_line_indxs(line: Geo, x_true, y_true, y_pred, threshold):
    losses = min_distance_loss(line, x_true, y_true, y_pred)
    return [i for i in range(len(x_true)) if losses[i] <= threshold]

def count_out_of_geo_error(line: Geo, x_true, y_true, y_pred, threshold):
    losses = min_distance_loss(line, x_true, y_true, y_pred)
    amount = [l for l in losses if l > threshold]
    return len(amount)


# Parabola loss and Error

def in_parabola_loss(parabola: Parabola, x_true, y_true, y_pred):
    pass


def distancia_minima_punto_parabola(punto, a, b, c):
    def distancia(x, y):
        return ((x - punto[0]) ** 2 + (y - punto[1]) ** 2) ** 0.5

    def parabola(x):
        return a * x ** 2 + b * x + c

    def distancia_entre_punto_y_parabola(x):
        return distancia(x, parabola(x))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(distancia_entre_punto_y_parabola)
    return res.fun


if __name__ == "__main__":
    line_ransac = RANSAC(
        n=2,
        t=1,
        d=2,
        k=500,
        model=Line(), 
        loss=min_distance_loss, 
        metric=count_out_of_geo_error
        )

    X = np.array([4, 5, 6, 6, 8, 9, 10, 20, 22, 23, 24, 25, 28, 30, 34, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 41])
    y = np.array([1113.2215576171875, 1113.3203125, 1113.5533447265625, 1007.476318359375, 1112.6373291015625, 966.0786743164062, 1111.1697998046875, 889.1231079101562, 968.5117797851562, 968.4758911132812, 965.6306762695312, 960.7706298828125, 938.2525634765625, 915.928466796875, 732.9209594726562, 953.2447509765625, 951.96044921875, 910.4817504882812, 1630.9703369140625, 951.2706909179688, 910.56982421875, 1630.9703369140625, 950.91357421875, 910.609375, 1631.0927734375, 951.1853637695312, 910.1572875976562, 952.0280151367188])

    import matplotlib.pyplot as plt

    line_ransac.fit(X, y)

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)

    plt.scatter(X, y)

    line = np.linspace(0, 50, num=100)
    plt.plot(line, line_ransac.predict(line), c="peru")
    plt.show()


    parabola_ransac = RANSAC(
        n=3,
        t=10,
        d=1,
        k=500,
        model=Parabola(), 
        loss=min_distance_loss, 
        metric=count_out_of_geo_error
        )
    
    parabola_ransac.fit(
        np.array(line_ransac.inlier_X), 
        np.array(line_ransac.inlier_y)
        )

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)

    plt.scatter(X, y)

    line = np.linspace(0, 50, num=100)
    plt.plot(line, parabola_ransac.predict(line), c="peru")
    plt.show()
