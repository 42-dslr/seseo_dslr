#!/usr/bin/python3

import sys
import pandas as pd
from load_csv import load
import matplotlib.pyplot as plt
import seaborn as sns
from logreg_train import data_fill


def main():
    try:
        data = load("datasets/dataset_train.csv")
        # data = load("datasets/dataset_test.csv")

        Gryffindor1 = ["Flying"]
        Gryffindor2 = ["Transfiguration"]
        Gryffindor3 = ["History of Magic"]
        
        Gryffindor = Gryffindor1 + Gryffindor2 + Gryffindor3
        print("Gryffindor features correlation:")
        print(data[Gryffindor].corr())
        print()

        Ravenclaw1 = ["Muggle Studies"]
        Ravenclaw2 = ["Charms"]

        Ravenclaw = Ravenclaw1 + Ravenclaw2
        print("Ravenclaw features correlation:")
        print(data[Ravenclaw].corr())
        print()
        
        Slytherin = ["Divination"]

        # Astronomy and Defense Against the Dark Arts -> correlated.
        Raven_Sly1 = ["Defense Against the Dark Arts"]
        Raven_Sly2 = ["Astronomy"]

        Raven_Sly = Raven_Sly1 + Raven_Sly2
        print("Ravenclaw - Slytherin features correlation:")
        print(data[Raven_Sly].corr())
        print()

        Raven_Gryffin = ["Ancient Runes"]
        Raven_Huffle = ["Herbology"]

        # data.dropna(inplace=True)
        data_fill(data)

        hog_house = "Hogwarts House"

        selected_feature = [hog_house] + Raven_Gryffin + Raven_Huffle + Raven_Sly1 + Gryffindor1 + Ravenclaw2 + Slytherin
        print("Selected features correlation:")
        print(data[selected_feature].corr(numeric_only=True))
        print()

        data_sample = data[selected_feature]
        print(data_sample)

        sns.pairplot(data=data_sample, hue=hog_house, diag_kind="hist", diag_kws=dict(multiple="stack"))
        plt.show()

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
