from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np
import csv

included_cols = [2,4,5,6,7,9,11]
included_cols2 = [1,3,4,5,6,8,10]

def getTrainingSet(file):
    X = []
    y = []
    firstcol = True;
    with open(file, newline='') as csvfile:
        res = csv.reader(csvfile, delimiter=',')
        for row in res:
            if (firstcol == True):
                firstcol = False
            else:
                row[1] = float(row[1])
                row[2] = int(row[2])
                if (row[4] == "male"):
                    row[4] = 0
                else:
                    row[4] = 1
                if (len(row[5]) == 0):
                    row[5] = np.nan
                else:
                    row[5] = float(row[5])
                row[6] = int(row[6])
                row[7] = int(row[7])
                row[9] = float(row[9])
                if (len(row[11]) == 0):
                    row[11] = np.nan
                if (row[11] == "S"):
                    row[11] = 1
                if (row[11] == "C"):
                    row[11] = 2
                if (row[11] == "Q"):
                    row[11] = 3
                X.append(row)
    y = [[l[i] for i in [1]] for l in X]
    y = [item for sublist in y for item in sublist]
    X = [[l[i] for i in included_cols] for l in X]
    return X, y

def getTestSet(file):
    X = []
    y = []
    firstcol = True;
    with open(file, newline='') as csvfile:
        res = csv.reader(csvfile, delimiter=',')
        for row in res:
            if (firstcol == True):
                firstcol = False
            else:
                row[1] = int(row[1])
                if (row[3] == "male"):
                    row[3] = 0
                else:
                    row[3] = 1
                if (len(row[4]) == 0):
                    row[4] = np.nan
                else:
                    row[4] = float(row[4])
                row[5] = int(row[5])
                row[6] = int(row[6])
                if (len(row[8]) == 0):
                    row[8] = np.nan
                else:
                    row[8] = float(row[8])
                if (len(row[10]) == 0):
                    row[10] = np.nan
                if (row[10] == "S"):
                    row[10] = 1
                if (row[10] == "C"):
                    row[10] = 2
                if (row[10] == "Q"):
                    row[10] = 3
                X.append(row)
    IDs = [[l[i] for i in [0]] for l in X]
    X = [[l[i] for i in included_cols2] for l in X]
    return X, IDs

X, y = getTrainingSet('train.csv')

ages = [[l[i] for i in [2]] for l in X]
embarked = [[l[i] for i in [6]] for l in X]
ages = [item for sublist in ages for item in sublist]
embarked = [item for sublist in embarked for item in sublist]
fare = [[l[i] for i in [5]] for l in X]
fare = [item for sublist in fare for item in sublist]

average_fare = np.nanmean(fare)
average_age = np.nanmean(ages)
most_common = max(set(embarked), key=embarked.count)

for r in X:
    if (np.isnan(r[2])):
        r[2] = average_age
    if (np.isnan(r[6])):
        r[6] = most_common

rfc = RandomForestClassifier(n_estimators=20, criterion="entropy");

##kf = KFold(len(X), n_folds=20)
##
##scores = []
##for train, test in kf:
##    rfc.fit([X[i] for i in train], [y[i] for i in train])
##    scores.append(rfc.score([X[i] for i in test], [y[i] for i in test]))
##print(np.mean(scores))

testX, IDs = getTestSet('test.csv')
IDs = [item for sublist in IDs for item in sublist]

for r in testX:
    if (np.isnan(r[2])):
        r[2] = average_age
    if (np.isnan(r[6])):
        r[6] = most_common
    if (np.isnan(r[5])):
        r[5] = average_fare

rfc.fit(X, y)
result = rfc.predict(testX)

pred = [list(a) for a in zip(IDs, result)]
with open('prediction.csv', 'w', newline='') as csvfile:
    wrt = csv.writer(csvfile, delimiter=',')
    for i in pred:
        wrt.writerow(i)
