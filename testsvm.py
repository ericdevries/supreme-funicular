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

connection = psycopg2.connect('host=192.168.1.244 dbname=instagram user=instagram password=instagram')
cursor = connection.cursor()

COLUMNS = [
    'followed_by',
    'likes',
    'comments',
    'engagement'
]

numpy.set_printoptions(linewidth=170, suppress=True)

def getdata():
    cursor.execute("""
        select u.followed_by, p.likes, p.comments, p.created_time
        from user_profiles u
        join posts p on p.user_id = u.id
        where u.followed_by > 100 and (p.likes / u.followed_by) < 1
        order by p.created_time desc
        limit 20000
    """)

    return cursor.fetchall()

def preprocess_row(row):
    row = list(row)

    followed_by = float(row[0])
    likes = float(row[1])
    comments = float(row[2])

    date = row[3]
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    diff = date - start

    row[3] = int(diff.total_seconds() / (5 * 60))

    row.append((likes + 10*comments) / followed_by )

    # strip userid and photoid
    return row


def main():
    data = getdata()
    data = list(map(preprocess_row, data))
    data = sorted(data, key=lambda x: x[3])

    array = numpy.array(data)
    array = array.astype(numpy.float)

    X = array[:,:-1]
    y = array[:,-1]

    times = X[:,3].reshape(-1, 1)
    times_divided = numpy.linspace(times.min(), times.max(), times.size) # no duplicate X values
    xnew = numpy.linspace(times_divided.min(), times_divided.max(), times.size / 12)
    xs = UnivariateSpline(times_divided, y)
    xs.set_smoothing_factor(10)

    svr_rbf = sklearn.svm.SVR(kernel='rbf', C=1, gamma=5)

    print('fitting rbf')
    y_rbf = svr_rbf.fit(times, y).predict(times)
    fig, ax1 = plt.subplots()

    ax1.plot(xnew, xs(xnew), color='green')
    ax2 = ax1.twinx()

    ax2.plot(times, y_rbf, color='red')

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
