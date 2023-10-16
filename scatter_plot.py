#!/usr/bin/python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from load_csv import load
from logreg_train import data_fill


def show_all(data: pd.DataFrame, lst: list[str]) -> None:

    print("selected feature list:", ", ".join(lst))

    feature_pair = list(combinations(lst, 2))

    x_max = 4
    y_max = len(feature_pair) // x_max + (1 if len(feature_pair) % x_max else 0)

    for y in range(y_max):
        _, axs = plt.subplots(1, x_max, figsize=(25, 7))
        for x in range(x_max):
            idx = x_max * y + x
            if idx < len(feature_pair):
                feature = list(feature_pair[idx])
                sns.scatterplot(data=data, x=feature[0], y=feature[1], hue="Hogwarts House", ax=axs[x])
        plt.show()
        print("graph data list:", feature_pair[y * x_max: (y + 1) * x_max])


def answer(data: pd.DataFrame, lst: list[str]) -> None:

    print("answer feature:", ", ".join(lst))

    sns.scatterplot(data=data, x=lst[0], y=lst[1], hue="Hogwarts House")
    plt.show()


def main():
    try:
        data = load("datasets/dataset_train.csv")
        # data = load("datasets/dataset_test.csv")

        # data.dropna(inplace=True)
        data_fill(data)

        # non_numeric = ["First Name", "Last Name", "Birthday", "Best Hand"]
        # exclude = ["Arithmancy", "Care of Magical Creatures"]

        # selected_feature = sorted(list(set(data.columns[2:]) - set(non_numeric) - set(exclude)))
        # show_all(data, selected_feature)

        # print("\n---------------------------------------------------\n")

        answer(data, ['Astronomy', 'Defense Against the Dark Arts'])
        print(data[['Astronomy', 'Defense Against the Dark Arts']].corr())

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
