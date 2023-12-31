#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import copy
import pickle
from load_csv import load, get_cur_dir

EPS = 1e-7

def sigmoid(z: np.ndarray) -> np.ndarray:
    """sigmoid function."""

    # for preventing overflow
    z_clipped = np.clip(z, -32, 32)
    return 1 / (np.exp(-z_clipped) + 1)


# def softmax(z: np.ndarray) -> np.ndarray:
#     """softmax function."""

#     exp_z = np.exp(z - np.max(z))
#     return exp_z / np.sum(exp_z, axis=0)


def get_one_hot_value(x: np.ndarray) -> np.ndarray:
    """one hot value for categorical. Max value -> 1, and the others -> 0."""

    max_idx = np.argmax(x, axis=1)
    res = np.zeros_like(x)
    for i in range(len(max_idx)):
        res[i, max_idx[i]] = 1
    return res


def one_hot_encoding(x: pd.Series) -> np.ndarray:
    """convert categorical variable to binary vector."""

    unique_val = x.unique()
    unique_val.sort()
    return (x.values.reshape(-1, 1) == unique_val).astype(int)


def propagate(w: np.ndarray, b: np.ndarray,
              X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)

    Return
        dw: gradient loss of weights
        db: gradient loss of bias
        cost: negative log-likelihood cost for logistic regression
    """

    # m: data size(n_data)
    m = X.shape[0]

    # A: Predicted value(Y_hat). (n_data, n_category)
    A = sigmoid(X @ w.T + b)

    # Cost(loss) add epsilon for preventing errors
    cost = - np.sum(Y * np.log(A + EPS) + (1 - Y) * np.log(1 - A + EPS)) / m

    dw = (A - Y).T @ X / m
    db = np.sum((A - Y).T, axis=1) / m

    cost = np.squeeze(np.array(cost))

    return dw, db, cost


def GD_optimizer(w: np.ndarray, b: np.ndarray,
                 X: np.ndarray, Y: np.ndarray,
                 epoch: int = 1,
                 lr: float = 0.001,
                 print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)
        epoch: number of iterations
        lr: learning rate
        print_cost: printing the loss every given steps
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    return BGD_optimizer(w, b, X, Y, epoch=epoch, batch=X.shape[0], lr=lr, print_cost=print_cost)


def SGD_optimizer(w: np.ndarray, b: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  epoch: int = 1,
                  lr: float = 0.001,
                  print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)
        epoch: number of iterations
        lr: learning rate
        print_cost: printing the loss every given steps
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    return BGD_optimizer(w, b, X, Y, epoch=epoch, batch=1, lr=lr, print_cost=print_cost)


def BGD_optimizer(w: np.ndarray, b: np.ndarray,
                  X: np.ndarray, Y: np.ndarray,
                  epoch: int = 1,
                  batch: int = 32,
                  lr: float = 0.001,
                  print_cost: int = 0) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)
        Y: train data Y (n_data, n_category)
        epoch: number of iterations
        batch: data size for updating w, b
        lr: learning rate
        print_cost: printing the loss every given steps
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    lim = len(X) // batch + (1 if len(X) % batch else 0)

    for i in range(epoch):

        # batch loop
        for j in range(lim):
        
            # calculate dw, db, cost
            dw, db, cost = propagate(w, b, X[batch * j: batch * (j + 1)], Y[batch * j: batch * (j + 1)])

            # update w, b
            w = w - lr * dw
            b = b - lr * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % print_cost == 0:
            print (f"The cost of iteration {i}: {cost}")

        # # check converged
        # if (np.isclose(dw, np.zeros_like(dw), rtol=1e-6).all() or
        #     np.isclose(db, np.zeros_like(db), rtol=1e-6).all()):
        #     break

    costs.append(cost)

    return w, b, costs


def predict(w: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Args
        w: weights (n_category, n_feature)
        b: bias (n_category,)
        X: train data X (n_data, n_feature)

    Return
        Y_pred: predict value (n_data, n_category)
    """

    m = X.shape[0]

    A = sigmoid(X @ w.T + b)

    if w.shape[0] == 1:
        Y_pred = (A > 0.5).astype(int)
        # for i in range(A.shape[1]):
        #     Y_pred[i, 0] = 1 if A[i, 0] > 0.5 else 0
    else:
        Y_pred = get_one_hot_value(A)

    return Y_pred.astype(int)


def model(x_train: np.ndarray, y_train: np.ndarray,
          w_init: np.ndarray = None, b_init: np.ndarray = None,
          epoch: int = 1,
          batch: int = 32,
          lr: int = 0.001,
          print_cost: int = 100,
          optimizer: str = "BGD") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Args
        x_train: train data for input (n_data, n_feature)
        y_train: train data for output (n_data, n_category)
        w_init: init w (n_category, n_feature)
        b_init: inti bias (n_category,)
        epoch: number of iterations
        batch: data size for updating w, b
        lr: learning rate
        print_cost: printing the cost every n steps. 0 or None for not printing.
    
    Return
        tuple(w, b, costs)
        costs: cost list of every 100 steps
    """

    assert len(y_train.shape) in (1, 2), "invalid y value."
    assert optimizer in ["BGD", "SGD", "GD"], "invalid optimizer."

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)

    n_feature = x_train.shape[1]
    n_category = y_train.shape[1]

    # w shape is (n_category, n_feature), and b shape is (n_category,)
    if w_init is None:
        w_init = np.zeros((n_category, n_feature))
    if b_init is None:
        b_init = np.zeros(n_category)

    match optimizer:
        case "BGD":
            w, b, costs = BGD_optimizer(w_init, b_init, x_train, y_train, epoch, batch, lr, print_cost)
        case "SGD":
            w, b, costs = SGD_optimizer(w_init, b_init, x_train, y_train, epoch, lr, print_cost)
        case "GD":
            w, b, costs = GD_optimizer(w_init, b_init, x_train, y_train, epoch, lr, print_cost)

    return w, b, costs


def get_feature_data(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """return x(features), y(answers)."""

    # Astronomy and Defense Against the Dark Arts -> correlated.
    exclude_list = ["Defense Against the Dark Arts", "Arithmancy", "Care of Magical Creatures", "Potions"]

    Gryffindor1 = ["Flying"]
    Gryffindor2 = ["Transfiguration"]
    Gryffindor3 = ["History of Magic"]

    Ravenclaw1 = ["Muggle Studies"]
    Ravenclaw2 = ["Charms"]

    Slytherin = ["Divination"]
    Ravenclaw = Ravenclaw1 + Ravenclaw2
    Gryffindor = Gryffindor1 + Gryffindor2 + Gryffindor3

    # correlated data
    Raven_Sly1 = ["Astronomy"]
    Raven_Sly2 = ["Defense Against the Dark Arts"]

    Raven_Huffle = ["Herbology"]
    Raven_Gryffin = ["Ancient Runes"]
    Raven_Sly = Raven_Sly1 + Raven_Sly2

    selected_feature = []
    selected_feature += Raven_Gryffin + Raven_Sly2 + Raven_Huffle
    # selected_feature += Gryffindor3 + Gryffindor2 + Gryffindor1
    selected_feature += Gryffindor1
    selected_feature += Slytherin
    # selected_feature += Ravenclaw1 + Ravenclaw2
    selected_feature += Ravenclaw2
    selected_feature = list(set(selected_feature))

    return data[selected_feature], selected_feature


def evaluate(w: np.ndarray, b: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """return accuracy of prediction."""
    
    y_pred = predict(w, b, x_test)
    
    assert y_pred.shape == y_test.shape, "invalid input. different shape"

    wrong = np.sum(np.abs(y_pred - y_test) / 2)
    
    return (1 - wrong / len(y_test)) * 100


def data_fill(data: pd.DataFrame) -> None:
    data["Defense Against the Dark Arts"].fillna(- data["Astronomy"] / 100, inplace=True)

    data.fillna(data.mean(numeric_only=True), inplace=True)


def main():
    try:
        assert len(sys.argv) == 2, "usage: python3 logreg_train.py [file name for train]."

        path = sys.argv[1]
        data: pd.DataFrame = load(path)
        data: pd.DataFrame = load("datasets/dataset_train.csv")

        assert data is not None, "data load failure."

        selected_feature = None

        path = get_cur_dir() + "/.param.pkl"
        try:
            with open(path, "rb") as f:
                w, b, selected_feature, mean, std = pickle.load(f)
        except: None

        sample_data = data

        if selected_feature is None:
            selected_df, selected_feature = get_feature_data(sample_data)
        else:
            selected_df = data[selected_feature]

        # fill NaN to mean value.
        x = selected_df.fillna(selected_df.mean(numeric_only=True))

        mean = x.mean(numeric_only=True)
        std = x.std(numeric_only=True)

        # standardize input (using Z score)
        x_train = ((x - mean) / std).to_numpy()
        y_train = one_hot_encoding(sample_data["Hogwarts House"])

        with open(path, "wb") as f:
            w, b, _ = model(x_train, y_train, epoch=10000, lr=0.005, print_cost=0, optimizer="GD")
            # w, b, _ = model(x_train, y_train, epoch=2000, batch=50, lr=0.005, print_cost=0, optimizer="BGD")
            # w, b, _ = model(x_train, y_train, epoch=40, lr=0.005, print_cost=0, optimizer="SGD")
            pickle.dump([w, b, selected_feature, mean, std], f)
            print(f"Param(w, b, selected_feature, mean, std) saved to {path}")

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
