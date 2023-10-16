#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
import time
from load_csv import load
from logreg_train import get_feature_data, one_hot_encoding, model, evaluate, data_fill
from tensorflow import keras


def main():

    try:

        data: pd.DataFrame = load("datasets/dataset_train.csv")

        assert data is not None, "data load failure."

        gdacc_k = []
        sgdacc_k = []
        bgdacc_k = []
        gdacc = []
        sgdacc = []
        bgdacc = []

        time_rec = []
        time_rec_k = []

        data_fill(data)

        for i in range(10):
            data_sample = data.sample(frac=1)

            selected_df, selected_feature = get_feature_data(data_sample)

            # fill NaN to mean value.
            x = selected_df.fillna(selected_df.mean())
            mean = x.mean()
            std = x.std()

            x = ((x - mean) / std).to_numpy()
            y = one_hot_encoding(data_sample["Hogwarts House"])

            x_train = x[:1400]
            y_train = y[:1400]

            x_test = x[1400:]
            y_test = y[1400:]

            gd_epoch = 8000
            sgd_epoch = 25
            bgd_epoch = 1500

            t = []
            t_k = []
            l = []
            l_k = []

            # gdmodel = keras.Sequential([
            #     keras.layers.Input(shape=(x_train.shape[1],)),
            #     keras.layers.Dense(y_test.shape[1], activation='sigmoid')
            # ])
            # gdoptimizer = keras.optimizers.SGD(learning_rate=0.005)
            # gdmodel.compile(optimizer=gdoptimizer,
            #             loss='categorical_crossentropy',
            #             metrics=['accuracy'])
            # t_s = time.time()
            # gdmodel.fit(x_train, y_train, epochs=gd_epoch, batch_size=x_train.shape[0], verbose=0)
            # t_e = time.time()
            # loss, accuracy = gdmodel.evaluate(x_test, y_test, verbose=0)
            # gdacc_k.append(round(accuracy * 100, 2))
            # t_k.append(t_e - t_s)

            # sgdmodel = keras.Sequential([
            #     keras.layers.Input(shape=(x_train.shape[1],)),
            #     keras.layers.Dense(y_test.shape[1], activation='sigmoid')
            # ])
            # sgdoptimizer = keras.optimizers.SGD(learning_rate=0.005)
            # sgdmodel.compile(optimizer=sgdoptimizer,
            #             loss='categorical_crossentropy',
            #             metrics=['accuracy'])
            # t_s = time.time()
            # sgdmodel.fit(x_train, y_train, epochs=sgd_epoch, batch_size=1, verbose=0)
            # t_e = time.time()
            # loss, accuracy = sgdmodel.evaluate(x_test, y_test, verbose=0)
            # sgdacc_k.append(round(accuracy * 100, 2))
            # t_k.append(t_e - t_s)

            # bgdmodel = keras.Sequential([
            #     keras.layers.Input(shape=(x_train.shape[1],)),
            #     keras.layers.Dense(y_test.shape[1], activation='sigmoid')
            # ])
            # bgdoptimizer = keras.optimizers.SGD(learning_rate=0.005)
            # bgdmodel.compile(optimizer=bgdoptimizer,
            #             loss='categorical_crossentropy',
            #             metrics=['accuracy'])
            # t_s = time.time()
            # bgdmodel.fit(x_train, y_train, epochs=bgd_epoch, batch_size=50, verbose=0)
            # t_e = time.time()
            # loss, accuracy = bgdmodel.evaluate(x_test, y_test, verbose=0)
            # bgdacc_k.append(round(accuracy * 100, 2))
            # t_k.append(t_e - t_s)


            t_s = time.time()
            w, b, costs = model(x_train, y_train, epoch=gd_epoch, lr=0.005, print_cost=0, optimizer="GD")
            t_e = time.time()
            gdacc.append(evaluate(w, b, x_test, y_test))
            t.append(t_e - t_s)

            t_s = time.time()
            w, b, costs = model(x_train, y_train, epoch=sgd_epoch, lr=0.005, print_cost=0, optimizer="SGD")
            t_e = time.time()
            sgdacc.append(evaluate(w, b, x_test, y_test))
            t.append(t_e - t_s)
            
            t_s = time.time()
            w, b, costs = model(x_train, y_train, epoch=bgd_epoch, batch=50, lr=0.005, print_cost=0, optimizer="BGD")
            t_e = time.time()
            bgdacc.append(evaluate(w, b, x_test, y_test))
            t.append(t_e - t_s)

            time_rec_k.append(tuple(t_k))
            time_rec.append(tuple(t))
            # print(f'loop {i} accuracy: {gdacc_k[-1]}%, {sgdacc_k[-1]}%, {bgdacc_k[-1]}%, {gdacc[-1]}%, {sgdacc[-1]}%, {bgdacc[-1]}%')
            print(f'loop {i} accuracy: {gdacc[-1]}%, {sgdacc[-1]}%, {bgdacc[-1]}%')

        gdacc = np.array(gdacc)
        sgdacc = np.array(sgdacc)
        bgdacc = np.array(bgdacc)
        gdacc_k = np.array(gdacc_k)
        sgdacc_k = np.array(sgdacc_k)
        bgdacc_k = np.array(bgdacc_k)

        time_rec = np.array(time_rec)
        time_rec_k = np.array(time_rec_k)

        # print("\n------ keras -------")

        # print(f"training time: {time_rec_k}")
        # print(f"gd_k mean: {gdacc_k.mean()}")
        # print(f"gd_k min: {gdacc_k.min()}")
        # print(f"gd_k max: {gdacc_k.max()}")
        # print(f"sgd_k mean: {sgdacc_k.mean()}")
        # print(f"sgd_k min: {sgdacc_k.min()}")
        # print(f"sgd_k max: {sgdacc_k.max()}")
        # print(f"bgd_k mean: {bgdacc_k.mean()}")
        # print(f"bgd_k min: {bgdacc_k.min()}")
        # print(f"bgd_k max: {bgdacc_k.max()}")

        print("\n------ my logic ------")

        print(f"training time: {time_rec}")
        print(f"gd mean: {gdacc.mean()}")
        print(f"gd min: {gdacc.min()}")
        print(f"gd max: {gdacc.max()}")
        print(f"sgd mean: {sgdacc.mean()}")
        print(f"sgd min: {sgdacc.min()}")
        print(f"sgd max: {sgdacc.max()}")
        print(f"bgd mean: {bgdacc.mean()}")
        print(f"bgd min: {bgdacc.min()}")
        print(f"bgd max: {bgdacc.max()}")

    
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()