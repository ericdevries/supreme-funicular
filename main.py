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
from data import data as dataeelco
from matplotlib import pyplot as plt

connection = psycopg2.connect('dbname=instagram user=instagram password=instagram')
cursor = connection.cursor()

COLUMNS = [
    'userid',
    'photoid',
    'follows',
    'followed_by',
    'type',
    'likes',
    'comments',
    'filter',
    'tags',
    'engagement'
]

numpy.set_printoptions(linewidth=170, suppress=True)


def plot_data_per_interval():

    xvalues = []
    yvalues = []

    with open('15minutesweek.csv', 'r') as f:
        reader = csv.reader(f)

        alldata = []

        for row in reader:
            alldata.append([int(row[0]), float(row[1]) / float(row[2])])

        alldata = sorted(alldata, key=lambda x: x[0])

        xvalues = [x[0] for x in alldata]
        yvalues = [y[1] for y in alldata]

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-0.1, 0.1,))
    yvalues = scaler.fit_transform(numpy.array(yvalues))

    smooth = UnivariateSpline(xvalues, yvalues)
    smooth.set_smoothing_factor(0)

    xnew = numpy.linspace(0, len(xvalues) - 1, 7 * 24)

    plt.plot(xnew, smooth(xnew), linestyle='--')

    e = []

    for d in dataeelco:
        for d2 in d:
            e.append(d2)

    smooth3 = UnivariateSpline(xvalues, e)
    smooth3.set_smoothing_factor(0.5)

    plt.plot(xnew, smooth3(xnew))

    plt.show()




def group_data_per_interval(data):
    results = {
        # 'time': { total: 123, posts: 34 }
    }

    from dateutil.tz import tzlocal
    localtz = tzlocal()

    for row in data:
        date = row[9]
        date = date.replace(tzinfo=pytz.utc)
        start = date - datetime.timedelta(days=date.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        diff = date - start

        timeslot = int(diff.total_seconds() / (15 * 60))
        followed_by = float(row[3])
        likes = float(row[5])
        comments = float(row[6])
        engagement = (likes + 10*comments) / followed_by

        if timeslot not in results:
            results[timeslot] = { 'total': 0, 'posts': 0 }

        results[timeslot]['total'] += engagement
        results[timeslot]['posts'] += 1

        #print([timeslot, engagement])


    with open('15minutesweek.csv', 'w') as f:
        writer = csv.writer(f)

        for k, v in results.items():
            print([k, v['total'], v['posts']])
            writer.writerow([k, v['total'], v['posts']])




def getdata():
    cursor.execute("""
        select u.id, p.id, u.follows, u.followed_by, p.type, p.likes, p.comments, p.filter, p.tags, p.created_time
        from user_profiles u
        join posts p on p.user_id = u.id
        where u.followed_by > 100
    """)

    return cursor.fetchall()

def preprocess_row(row):
    row = list(row)

    # rows: userid, photoid, follows, followed_by, type, likes, comments, filter, tags
    if row[4] == 'image':
        row[4] = 0
    else:
        row[4] = 1

    row[8] = len(row[8].split(' '))

    followed_by = float(row[3])
    likes = float(row[5])
    comments = float(row[6])

    date = row[9]
    #start = date - datetime.timedelta(days=date.weekday())
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    diff = date - start

    row[9] = int(diff.total_seconds() / (15 * 60))

    row.append((likes + 10*comments) / followed_by)

    # strip userid and photoid
    return row[2:]


def generate_engagement_trends(engagement, times):
    results = {}
    data = zip(times, engagement)

    for t, v in data:
        if t not in results:
            results[t] = (0, 0.0,)

        results[t] = (results[t][0] + 1, results[t][1] + v,)

    summarized = []

    for k, v in results.items():
        summarized.append((k, v[1] / v[0]))

    summarized = sorted(summarized, key=lambda item: item[0])

    keys = [k[0] for k in summarized]
    values = [k[1] for k in summarized]

    return numpy.array(keys), numpy.array(values)


def plot_engagement(keys, values):
    xnew = numpy.linspace(keys.min(), keys.max(), 7 * 24)
    smooth = spline(keys, values, xnew)

    plt.plot(xnew, smooth)
    plt.show()




def main():
    plot_data_per_interval()
    return


    data = getdata()
    grouped = group_data_per_interval(data)

    data = list(map(preprocess_row, data))
    data = sorted(data, key=lambda x: x[7])

    array = numpy.array(data)

    # category features
    enc = sklearn.preprocessing.LabelEncoder()
    enc.fit(array[:,5])
    array[:,5] = enc.transform(array[:,5])

    array = array.astype(numpy.float)

    times = array[:,7]

    grouped = group_data_per_interval(data)

    return

    # everything between 0 and 1
    scaler = sklearn.preprocessing.MinMaxScaler()
    array = scaler.fit_transform(array)

    #trends_keys, trends_values = generate_engagement_trends(array[:,-1], times)

    #array = sklearn.preprocessing.scale(array)

    X = array[:,:-1]
    y = array[:,-1]
    # do model selection stuff

    poly = sklearn.preprocessing.PolynomialFeatures(3)
    X = poly.fit_transform(X)



    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4, random_state=0)

    clf = sklearn.linear_model.SGDRegressor()
    fit = clf.fit(X_train, y_train)
    results = fit.predict(X_test)

    score = fit.score(X_test, y_test)

    scaled_times = X_test[:,7]
    total = sorted(zip(scaled_times, results, y_test), key=lambda x: x[0])

    sorted_times = numpy.array([x[0] for x in total])
    sorted_results = numpy.array([x[1] for x in total])
    sorted_ytest = numpy.array([x[2] for x in total])

    sorted_times = numpy.array(range(0, len(sorted_times)))

    xnew = numpy.linspace(sorted_times.min(), sorted_times.max(), 24)

    smooth_results = UnivariateSpline(sorted_times, sorted_results)
    smooth_results2 = UnivariateSpline(sorted_times, sorted_results)
    smooth_ytest = UnivariateSpline(sorted_times, sorted_ytest)
    smooth_results.set_smoothing_factor(0.5)
    #smooth_ytest.set_smoothing_factor(200)

    #smooth_results = spline(sorted_times, sorted_results, xnew)
    #smooth_ytest = spline(sorted_times, sorted_ytest, xnew)

    plt.plot(xnew, smooth_results(xnew),
             xnew, smooth_results2(xnew))
             #xnew, smooth_ytest(xnew))

    plt.show()



    #plot_engagement(trends_keys, trends_values)


if __name__ == '__main__':
    main()
