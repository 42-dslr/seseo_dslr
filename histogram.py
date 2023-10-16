#!/usr/bin/python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_csv import load
from logreg_train import data_fill


def show_non_numeric(data: pd.DataFrame, lst: list[str]) -> None:

    print("non_numeric data list:", ", ".join(lst))
    x_max = len(lst)
    _, axs = plt.subplots(1, x_max, figsize=(25, 10))
    for x in range(x_max):
        sns.histplot(data=data, x=lst[x], hue="Hogwarts House", multiple="stack", ax=axs[x])
    plt.show()
    print("graph data list:", ", ".join(lst))


def show_numeric(data: pd.DataFrame, lst: list[str]) -> None:

    print("numeric data list:", ", ".join(lst))
    y_max = 5
    x_max = 3
    for y in range(y_max):
        _, axs = plt.subplots(1, x_max, figsize=(25, 10))
        for x in range(x_max):
            idx = x_max * y + x
            if idx < len(lst):
                sns.histplot(data=data, x=lst[idx], hue="Hogwarts House", multiple="stack", ax=axs[x])
        plt.show()
        print("graph data list:", ", ".join(lst[y * x_max: (y + 1) * x_max]))


def answer(data: pd.DataFrame, lst: list[str]) -> None:

    print("answer feature:", ", ".join(lst))
    x_max = len(lst)
    _, axs = plt.subplots(1, x_max, figsize=(25, 10))
    for x in range(x_max):
        sns.histplot(data=data, x=lst[x], hue="Hogwarts House", multiple="stack", ax=axs[x])
    plt.show()


def main():
    try:
        data = load("datasets/dataset_train.csv")
        assert data is not None, "no data."

        # data.dropna(inplace=True)
        data_fill(data)

        # non_numeric = ["First Name", "Last Name", "Birthday", "Best Hand"]
        # numeric = sorted(list(set(data.columns[2:]) - set(non_numeric)))

        # show_non_numeric(data, non_numeric)

        # print("\n---------------------------------------------------\n")

        # show_numeric(data, numeric)

        # print("\n---------------------------------------------------\n")

        answer(data, ["Arithmancy", "Care of Magical Creatures"])

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
