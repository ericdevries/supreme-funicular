import psycopg2
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import sklearn.feature_selection
import sklearn.linear_model

from scipy.interpolate import spline
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline

import numpy
import datetime
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


def getdata():
    cursor.execute("""
        select u.id, p.id, u.follows, u.followed_by, p.type, p.likes, p.comments, p.filter, p.tags, p.created_time
        from user_profiles u
        join posts p on p.user_id = u.id
        where u.followed_by > 100
        limit 500000
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
    likes = float(row[4])
    comments = float(row[5])

    date = row[9]
    #start = date - datetime.timedelta(days=date.weekday())
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    diff = date - start

    row[9] = int(diff.total_seconds() / (5 * 60))

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
    data = getdata()
    data = list(map(preprocess_row, data))
    data = sorted(data, key=lambda x: x[7])

    array = numpy.array(data)

   # print(array)

    # category features
    enc = sklearn.preprocessing.LabelEncoder()
    enc.fit(array[:,5])
    array[:,5] = enc.transform(array[:,5])

    array = array.astype(numpy.float)

    times = array[:,7]

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

    print('training data', X_train.shape, y_train.shape)
    print('test data', X_test.shape, y_test.shape)

    clf = sklearn.linear_model.SGDRegressor()
    fit = clf.fit(X_train, y_train)
    results = fit.predict(X_test)

    score = fit.score(X_test, y_test)

    print(score)
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
