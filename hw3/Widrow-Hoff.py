import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import implements_the_modeling as mod
import sklearn

def Widrow_Hoff(x_train: pd.DataFrame, y_train: pd.DataFrame, nbIterations: int, n: int):
    x_train, y_train = x_train.values, y_train.values
    x_train = sklearn.preprocessing.MinMaxScaler().fit_transform(x_train)

    W = np.zeros(x_train.shape[1])

    for i in range(nbIterations):
        index = np.random.randint(low=0, high=x_train.shape[0])
        Xt = X[index, :]
        Yt = Y[index]

        #y_pred = W*Xt

        W = W - n*(Xt.dot(W)-Yt).dot(Xt)


    return W


def main():
    return 0

if __name__ == '__main__':
    main()