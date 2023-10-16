#!/usr/bin/python3

import sys
import pandas as pd
from load_csv import load
from typing import Any


def describe(data: pd.DataFrame, include=None, exclude=None) -> pd.DataFrame:
    '''This function returns described data.'''

    if not isinstance(data, pd.DataFrame):
        raise TypeError("invalid input type.")

    if data.size == 0:
        raise ValueError("Empty data.")

    new_data = data
    if include is not None or exclude is not None:
        new_data = data.select_dtypes(include=include, exclude=exclude)

    if len(new_data.columns) == 0:
        raise ValueError("No data.")

    empty_option = ["count"]
    obj_option = ["count", "unique", "top", "freq"]
    num_option = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    all_option = ["count", "unique", "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max"]

    ret = pd.DataFrame()
    for col in new_data.columns:
        column = new_data[col]
        if column.dtype in ["int64", "float64"]:
            column.dropna(inplace=True)
            
            if column.size:
                stats = [
                    column.size,
                    ft_mean(column),
                    ft_std(column),
                    ft_min(column),
                    ft_quartile(column, 0.25),
                    ft_quartile(column, 0.5),
                    ft_quartile(column, 0.75),
                    ft_max(column)
                ]

                tmp = pd.DataFrame(stats, index=num_option, columns=[col])

            else:
                stats = [0]
                tmp = pd.DataFrame(stats, index=empty_option, columns=[col])

        else:
            unique_lst = list(column.unique())

            cnt_dict = dict(zip(unique_lst, [0] * len(unique_lst)))
            for elem in column:
                cnt_dict[elem] += 1

            most_elem = unique_lst[0]
            most_elem_cnt = cnt_dict[most_elem]
            for k, v in cnt_dict.items():
                if most_elem_cnt < v:
                    most_elem_cnt = v
                    most_elem = k

            stats = [
                len(column),
                len(unique_lst),
                most_elem,
                most_elem_cnt
            ]
            tmp = pd.DataFrame(stats, index=obj_option, columns=[col])

        ret = ret.merge(tmp, left_index=True, right_index=True, how='outer')

    if len(ret.index) == len(all_option):
        ret = ret.reindex(all_option)

    elif len(ret.index) == len(num_option):
        ret = ret.reindex(num_option)

    return ret


def ft_mean(args: Any) -> float:
    """return mean values from given args."""
    if (
        len(args) == 0 or
        not all(map(lambda x: isinstance(x, (int, float)), args))
    ):
        raise ValueError("ft_mean")
    return sum(args) / len(args)


def ft_quartile(args: Any, p: float) -> list[float]:
    """return quartile values(25%, 75%) from given args."""
    if not isinstance(p, (int, float)):
        raise TypeError("Please check input type.")

    if (
        len(args) == 0 or
        not all(map(lambda x: isinstance(x, (int, float)), args)) or
        p > 1 or p < 0
    ):
        raise ValueError("Please check input value.")

    args = sorted(args)
    if len(args) == 1:
        return args[0]

    vidx = p * (len(args) - 1) + 1
    i = int(vidx)

    return args[i - 1] + (args[i] - args[i - 1]) * (vidx - i)


def ft_std(args: Any) -> float:
    """return standard deviation value from given args."""
    if (
        len(args) == 0 or
        not all(map(lambda x: isinstance(x, (int, float)), args))
    ):
        raise ValueError("ft_std")
    return ft_var(args) ** 0.5


def ft_var(args: Any) -> float:
    """return variance value from given args."""
    if (
        len(args) == 0 or
        not all(map(lambda x: isinstance(x, (int, float)), args))
    ):
        raise ValueError("ft_var")
    m = ft_mean(args)
    tmp = [(m - v) ** 2 for v in args]

    return sum(tmp) / (len(tmp) - 1)


def ft_min(args: Any) -> float:
    """return min value from given args."""
    if (
        len(args) == 0 or
        not all(map(lambda x: isinstance(x, (int, float)), args))
    ):
        raise ValueError("ft_min")

    v = args[0]
    for arg in args:
        if v > arg:
            v = arg
    return v


def ft_max(args: Any) -> float:
    """return max value from given args."""
    if (
        len(args) == 0 or
        not all(map(lambda x: isinstance(x, (int, float)), args))
    ):
        raise ValueError("ft_max")

    v = args[0]
    for arg in args:
        if v < arg:
            v = arg
    return v


def main():

    try:
        data = load("./datasets/dataset_train.csv")
        data = load("./datasets/dataset_test.csv")

        print("Describe all data:")
        print(describe(data))
        print()

        print("Describe numeric data:")
        print(describe(data, include='number'))
        print()

        print("Describe non numeric data:")
        print(describe(data, exclude='number'))

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()