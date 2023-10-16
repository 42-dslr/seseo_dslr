#!/usr/bin/python3

import sys
import pandas as pd
from load_csv import load
from describe import describe


def main():

    try:
        data = load("./datasets/dataset_train.csv")

        print("Describe all data:")
        print(describe(data).to_string() == data.describe(include='all').to_string())
        print()

        print("Describe numeric data:")
        print(describe(data, include='number').to_string() == data.describe(include='number').to_string())
        print()

        print("Describe non numeric data:")
        print(describe(data, exclude='number').to_string() == data.describe(exclude='number').to_string())

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()