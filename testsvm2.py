import psycopg2
import pytz
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import sklearn.feature_selection
import sklearn.linear_model

from scipy.interpolate import spline
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline

import numpy
import csv
import datetime
from matplotlib import pyplot as plt

COLUMNS = [
    'followed_by',
    'likes',
    'comments',
    'engagement'
]

numpy.set_printoptions(linewidth=170, suppress=True)

def getdata():
    rows = []

    with open('cows-data.csv') as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) != 2:
                continue

            if '-' not in row[0]:
                continue

            year = row[0].split('-')[0]
            month = row[0].split('-')[1]
            amount = row[1]

            rows.append((year, month, amount,))

    return rows

def preprocess_row(row):
    return [float(f) for f in row]


def main():
    data = getdata()

    array = numpy.array(data)
    array = array.astype(numpy.float)

    train = array[:100]
    test = array[100:]

    length = train.shape[0]
    test_length = test.shape[0]

    Xall = numpy.linspace(0, 168, 168).reshape(-1, 1) # no duplicate X values
    X = numpy.linspace(0, length, length).reshape(-1, 1) # no duplicate X values
    X_test = numpy.linspace(length, length + test_length, test_length).reshape(-1, 1) # no duplicate X values

    y = train[:,-1]
    y_test = test[:,-1]

    alltimedata = array[:,:-1]
    timedata = train[:,:-1]
    timedata_test = test[:,:-1]

    svr_rbf1 = sklearn.svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    fitted = svr_rbf1.fit(timedata, y)
    y_rbf1 = fitted.predict(alltimedata)

    fig, ax1 = plt.subplots()


    ax1.plot(X, y, color='green')
    ax1.plot(X_test, y_test, color='blue')
    ax1.plot(Xall, y_rbf1, color='red', linestyle='--')
    #ax1.plot(X, y_rbf2, color='yellow', linestyle='--')

    ax2 = ax1.twinx()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
