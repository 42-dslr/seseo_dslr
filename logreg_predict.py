#!/usr/bin/python3

import sys
import pickle
import pandas as pd
from load_csv import load, get_cur_dir
from logreg_train import predict


def main():
    """Estimating price program for ft_linear_regression."""

    try:
        assert len(sys.argv) == 2, "usage: python3 logreg_predict.py [test_file_name]."

        path = sys.argv[1]
        data = load(path)

        path = get_cur_dir() + "/.param.pkl"
        with open(path, "rb") as f:
            w, b, selected_feature, mean, std = pickle.load(f)

        answer_list = sorted(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])

        assert w.shape == (len(answer_list), len(selected_feature)), "invalid weight shape."

        # get Z score
        data.fillna(mean, inplace=True)
        x_test_tmp = data[selected_feature]
        x_test = ((x_test_tmp - mean) / std).to_numpy()

        y_pred = predict(w, b, x_test)
        
        with open(get_cur_dir() + "/houses.csv", "w") as f:
            f.write("Index,Hogwarts House\n")
            for idx, pred_idx in enumerate(y_pred.argmax(axis=1)):
                f.write(f"{idx},{answer_list[pred_idx]}\n")

    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
