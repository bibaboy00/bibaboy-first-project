from collections import defaultdict
import numpy as np
import urllib
import os
import datetime
import timeit
os.chdir('/Users/wangzhen/PycharmProjects/hw258/Assignment2')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random
random.seed(1)
import math

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield l.strip().split(',')

def tsdiff(t1, t2, bucket):
    if (t1 > t2):
        t1, t2 = t2, t1
    return (datetime.datetime.fromtimestamp(t2).date().year - datetime.datetime.fromtimestamp(t1).date().year) / bucket
start_ts = 942192000
bucket = 1
def preprocess(rating_data_time, start_ts):
    for i in range(len(rating_data_time)):
        for j in range(len(rating_data_time[i][1])):
            rating_data_time[i][1][j] = \
                (rating_data_time[i][1][j][0], rating_data_time[i][1][j][1],
                 (tsdiff(int(rating_data_time[i][1][j][2]), start_ts, bucket)))
    return rating_data_time
def timerange(rating_data_time):
    time_bucket = []
    for user, rates in rating_data_time:
        for _, _, time in rates:
            time_bucket.append(time)
    return (min(time_bucket), max(time_bucket))

data_train = list(parseData('ratings_train.csv'))
data_valid = list(parseData('ratings_validation.csv'))
data_test = list(parseData('ratings_test.csv'))

class BiasLFM(object):
    def __init__(self, rating_data, rating_data1, F, alpha=0.01, lmbd1=0.1, lmbd2=0.1,
                 max_iter=10, eps=10**(-3), decay=1.0):
        self.F = F
        self.P = defaultdict(list)
        self.Q = defaultdict(list)
        self.bu = defaultdict(list)
        self.bi = defaultdict(list)
        self.alpha = alpha
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.max_iter = max_iter
        self.rating_data_train = rating_data
        self.mu = sum([float(r[2]) for r in rating_data]) / len(rating_data)
        self.eps = eps
        self.decay = decay
        self.rating_data_valid = rating_data1

        for user, item, rate, _ in self.rating_data_train:
            self.P[user] = np.array([random.random() / math.sqrt(self.F)
                            for x in xrange(self.F)])
            self.Q[item] = np.array([random.random() / math.sqrt(self.F)
                            for x in xrange(self.F)])
            self.bu[user] = 0
            self.bi[item] = 0

    def train(self):
        start = timeit.default_timer()
        mse_old = self.mse(self.rating_data_train)
        for step in xrange(self.max_iter):
            print('step: ' + str(step) + ' mse_train: ' + str(mse_old)) + \
                 '  mse_valid: ' + str(self.mse(self.rating_data_valid))
            for user, item, rui, _, in self.rating_data_train:
                hat_rui = self.predict(user, item)
                err_ui = float(rui) - hat_rui
                pu = self.P[user]
                qi = self.Q[item]
                self.bu[user] += self.alpha * (err_ui - self.lmbd1 * self.bu[user])
                self.bi[item] += self.alpha * (err_ui - self.lmbd1 * self.bi[item])
                pu += self.alpha * (err_ui * qi - self.lmbd2 * pu)
                qi += self.alpha * (err_ui * pu - self.lmbd2 * qi)
                self.P[user] = pu
                self.Q[item] = qi
            mse_new = self.mse(self.rating_data_train)
            if (abs(mse_new - mse_old) < self.eps):
                break
            mse_old = mse_new
            self.alpha *= self.decay
        end = timeit.default_timer()
        print('train time: ' + str(end - start))

    def predict(self, user, item):
        if user in self.P and item in self.Q:
            return self.P[user].dot(self.Q[item]) + self.bu[user] + self.bi[item] + self.mu
        elif user in self.P:
            return self.bu[user] + self.mu
        elif item in self.Q:
            return self.bi[item] + self.mu
        else:
            return self.mu

    def mse(self, rating_data):
        error = 0
        num = 0
        for user, item, rui, _ in rating_data:
            hat_rui = self.predict(user, item)
            if hat_rui > 5: hat_rui = 5
            error += (hat_rui - float(rui)) ** 2
            num += 1
        return error / num

class SVDPP(object):
    def __init__(self, rating_data, rating_data1, F, alpha=0.01,
                 lmbd1=0.1, lmbd2=0.1, lmbd3=0.1, max_iter=10, eps=10**(-3), decay=1.0):
        self.F = F
        self.P = dict()
        self.Q = dict()
        self.Y = dict()
        self.bu = dict()
        self.bi = dict()
        self.alpha = alpha
        self.lmbd1= lmbd1
        self.lmbd2 = lmbd2
        self.lmbd3 = lmbd3
        self.max_iter = max_iter
        self.rating_data_train = rating_data
        self.mu = sum([float(r[2]) for r in rating_data]) / len(rating_data)
        self.rating_data_train_dict = defaultdict(set)
        self.eps = eps
        self.decay = decay
        self.rating_data_valid = rating_data1
        for user, item, _, _ in self.rating_data_train:
            self.rating_data_train_dict[user].add(item)

        for user, item, _, _ in self.rating_data_train:
            if user not in self.P:
                self.P[user] = np.array([random.random() / math.sqrt(self.F)
                                for x in xrange(self.F)])
            if item not in self.Q:
                self.Q[item] = np.array([random.random() / math.sqrt(self.F)
                                for x in xrange(self.F)])
                self.Y[item] = np.array([random.random() / math.sqrt(self.F)
                                for x in xrange(self.F)])
            self.bu[user] = 0
            self.bi[item] = 0

    def train(self):
        start = timeit.default_timer()
        mse_old = self.mse(self.rating_data_train)
        for step in xrange(self.max_iter):
            start0 = timeit.default_timer()
            print('step: ' + str(step) + ' mse: ' + str(mse_old)) + \
                 '  mse_valid: ' + str(self.mse(self.rating_data_valid))
            for user, item, rui, _ in self.rating_data_train:
                z = np.array([0.0 for f in xrange(self.F)])
                for item0 in self.rating_data_train_dict[user]:
                    z += np.array(self.Y[item0])
                ru = 1.0 / math.sqrt(len(self.rating_data_train_dict[user]))
                hat_rui = self.predict(user, item, self.rating_data_train_dict[user])
                err_ui = float(rui) - hat_rui
                self.bu[user] += self.alpha * (err_ui - self.lmbd1 * self.bu[user])
                self.bi[item] += self.alpha * (err_ui - self.lmbd1 * self.bi[item])
                pu = self.P[user]
                qi = self.Q[item]
                yi = self.Y[item]
                pu += self.alpha * (err_ui * qi - self.lmbd2 * pu)
                qi += self.alpha * (err_ui * (pu + z * ru) - self.lmbd2 * qi)
                yi += self.alpha * (err_ui * qi * ru - self.lmbd3 * yi)
                self.P[user] = pu
                self.Q[item] = qi
                self.Y[item] = yi
            mse_new = self.mse(self.rating_data_train)
            end0 = timeit.default_timer()
            print end0-start0
            if (abs(mse_new - mse_old) < self.eps):
                break
            mse_old = mse_new
            self.alpha *= self.decay
        end = timeit.default_timer()
        print('train time: ' + str(end - start))

    def predict(self, user, item, ratedItems):
        if user not in self.P and item not in self.Q:
            return self.mu
        elif item not in self.Q:
            return self.bu[user] + self.mu
        elif user not in self.P:
            return self.bi[item] + self.mu
        else:
            z = np.array([0.0 for f in xrange(self.F)])
            for i in ratedItems:
                z += self.Y[i]
            return (self.P[user] + z / math.sqrt(len(ratedItems))).dot(self.Q[item]) \
                    + self.bu[user] + self.bi[item] + self.mu

    def mse(self, rating_data):
        error = 0
        num = 0
        for user, item, rui, _ in rating_data:
            ratedItems = self.rating_data_train_dict[user]
            hat_rui = self.predict(user, item, ratedItems)
            if hat_rui > 5:
                hat_rui = 5
            error += (hat_rui - float(rui)) ** 2
            num += 1
        return error / num

class TimeLM(object):
    def __init__(self, rating_data, rating_data1,
                 alpha=0.01, lmbd=0.1, max_iter=10, eps=10**(-3), decay=1.0, t_bin=16):
        '''先preprocess rating_data变成list<(user,list<(item,rate,time)>)>类型'''
        self.bu = dict()
        self.bi = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data_train = rating_data
        self.rating_data_valid = rating_data1
        self.mu = 0.0
        self.rating_data_train_dict = defaultdict(list)
        self.eps = eps
        self.decay = decay
        self.T = t_bin

        cnt = 0
        for user, rates in self.rating_data_train:
            self.bu[user] = [0] * self.T
            cnt += len(rates)
            for item, rate, time in rates:
                self.mu += rate
                self.bi[item] = [0] * self.T
        self.mu /= cnt

    def train(self):
        '''随机梯度下降法训练参数bu和bi'''
        start = timeit.default_timer()
        mse_old = self.mse(self.rating_data_train)
        for step in xrange(self.max_iter):
            print('step: ' + str(step) + ' mse_train: ' + str(mse_old)) + \
                 '  mse_valid: ' + str(self.mse(self.rating_data_valid))
            for user, rates in self.rating_data_train:
                for item, rui, time in rates:
                    hat_rui = self.predict(user, item, time)
                    err_ui = rui - hat_rui
                    self.bu[user][time] += self.alpha * (err_ui - self.lmbd * self.bu[user][time])
                    self.bi[item][time] += self.alpha * (err_ui - self.lmbd * self.bi[item][time])
            mse_new = self.mse(self.rating_data_train)
            if (abs(mse_new - mse_old) < self.eps):
                break
            mse_old = mse_new
            self.alpha *= self.decay
        end = timeit.default_timer()
        print('train time: ' + str(end - start))

    def predict(self, user, item, time):
        if user not in self.bu and item not in self.bi:
            return self.mu
        elif user not in self.bu:
            return self.bi[item][time] + self.mu
        elif item not in self.bi:
            return self.bu[user][time] + self.mu
        return self.bu[user][time] + self.bi[item][time] + self.mu

    def mse(self, rating_data):
        error = 0
        num = 0
        for user, rates in rating_data:
            for item, rui, time in rates:
                hat_rui = self.predict(user, item, time)
                if hat_rui > 5: hat_rui = 5
                error += (hat_rui - rui) ** 2
                num += 1
        return error / num

class TimeLM1(object):
    def __init__(self, rating_data, rating_data1,
                 alpha=0.01, lmbd=0.1, max_iter=10, eps=10**(-3), decay=1.0, t_bin=16):
        '''先preprocess rating_data变成list<(user,list<(item,rate,time)>)>类型'''
        self.bu = dict()
        self.bi = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data_train = rating_data
        self.rating_data_valid = rating_data1
        self.rating_data_train_dict = defaultdict(list)
        self.eps = eps
        self.decay = decay
        self.T = t_bin
        self.mu = [0] * self.T

        cnt = [0] * self.T
        for user, rates in self.rating_data_train:
            for item, rate, time in rates:
                self.bu[user] = 0
                self.bi[item] = 0
                cnt[time] += 1
                self.mu[time] += rate
        for t in range(self.T):
            self.mu[t] /= cnt[t]

    def train(self):
        '''随机梯度下降法训练参数bu和bi'''
        start = timeit.default_timer()
        mse_old = self.mse(self.rating_data_train)
        for step in xrange(self.max_iter):
            print('step: ' + str(step) + ' mse_train: ' + str(mse_old)) + \
                 '  mse_valid: ' + str(self.mse(self.rating_data_valid))
            for user, rates in self.rating_data_train:
                for item, rui, time in rates:
                    hat_rui = self.predict(user, item, time)
                    err_ui = rui - hat_rui
                    self.bu[user] += self.alpha * (err_ui - self.lmbd * self.bu[user])
                    self.bi[item] += self.alpha * (err_ui - self.lmbd * self.bi[item])
            mse_new = self.mse(self.rating_data_train)
            if (abs(mse_new - mse_old) < self.eps):
                break
            mse_old = mse_new
            self.alpha *= self.decay
        end = timeit.default_timer()
        print('train time: ' + str(end - start))

    def predict(self, user, item, time):
        if user not in self.bu and item not in self.bi:
            return self.mu[time]
        elif user not in self.bu:
            return self.bi[item] + self.mu[time]
        elif item not in self.bi:
            return self.bu[user] + self.mu[time]
        return self.bu[user] + self.bi[item] + self.mu[time]

    def mse(self, rating_data):
        error = 0
        num = 0
        for user, rates in rating_data:
            for item, rui, time in rates:
                hat_rui = self.predict(user, item, time)
                if hat_rui > 5: hat_rui = 5
                error += (hat_rui - rui) ** 2
                num += 1
        return error / num


#F=0, lmbd=0.1, alpha=0.05, decay=0.7 1.0945

#F=2, lmbd1=0.1, lmbd2=2.0, alpha=0.05, decay=0.7  1.0942
#F=10, lmbd1=0.1, lmbd2=3.0, alpha=0.05, decay=0.7, 1.0943
biaslfm = BiasLFM(data_train, data_valid,
                  F=10, max_iter=200, lmbd1=0.1, lmbd2=3.0, alpha=0.05, decay=0.7)
biaslfm.train()
print(biaslfm.mse(data_valid))
print(biaslfm.mse(data_test))

#F=10, lmbd1=0.1, lmbd2=3.0, lmbd3=20.0, alpha=0.05, decay=0.7, eps=0.0001  1.0943
svdpp = SVDPP(data_train, data_valid,
              F=10, max_iter=200, lmbd1=0.1, lmbd2=3.0, lmbd3=10.0,  alpha=0.05, decay=0.7)
svdpp.train()
print(svdpp.mse(data_valid))
print(svdpp.mse(data_test))

#lmbd=0.1, alpha=0.05, decay=0.7, t_bin=16(year) 1.1845
#lmbd=0.1, alpha=0.05, decay=0.7, t_bin=179(year) 1.2843
timelm = TimeLM(data_train, data_valid,
                  max_iter=200, lmbd=0.1, alpha=0.05, decay=0.7, t_bin=179)
timelm.train()
print(timelm.mse(data_valid))
print(timelm.mse(data_test))

#t_bin=16 1.0959
#t_bin=8 1.0955
#t_bin=6 1.0955
#t_bin=4 1.0957
#t_bin=3 1.0957
#t_bin=2 1.0949 (8)

timelm1 = TimeLM1(data_train, data_valid,
                  max_iter=200, lmbd=0.1, alpha=0.05, decay=0.7, t_bin=16)
timelm1.train()
print(timelm1.mse(data_valid))
print(timelm1.mse(data_test))

### to prove time may just be noise ###
rating_year = defaultdict(list)
for user, rates in data_train:
    for _,rate,time in rates:
        rating_year[time].append(rate)
rating_year_mean = [sum(rating_year[time]) / len(rating_year[time]) for time in rating_year]
rating_year_var = [np.var(rating_year[time]) for time in rating_year]
rating_year_num = [len(rating_year[time]) for time in rating_year]

def plotTwo(xLabel, (yLabelL, yLabelR), xValue, (yValueL, yValueR)):
    fig = plt.figure(dpi = 100)
    plt.xlabel(xLabel)
    axLeft = fig.add_subplot(111)
    axRight = fig.add_subplot(111)
    axRight = axLeft.twinx()
    axLeft.plot(xValue, yValueL, c = "tab:blue", label = yLabelL);
    axLeft.set_ylabel(yLabelL)
    axLeft.legend(loc = 1)
    axRight.plot(xValue, yValueR, c = "tab:orange", label = yLabelR);
    axRight.set_ylabel(yLabelR)
    axRight.legend(loc = 2)
    plt.show()


### find a good K for SVD model ###
fold = 5
valid_error =[]
for k in range(21):
    error = 0
    for i in range(fold):
        l = len(data_train)
        valid1 = data_train[l/5*i: l/5*(i+1)]
        train1 = data_train[:l/5*i] + data_train[l/5*(i+1):]
        biaslfm = BiasLFM(train1, valid1,
                          F=k, max_iter=200, lmbd1=0.1, lmbd2=3.0, alpha=0.05, decay=0.7)
        biaslfm.train()
        error += biaslfm.mse(valid1)
    error /= fold
    valid_error.append(error)

valid_error_record = \
[1.2964437631599144, 1.2961784883873126, 1.2962954862130049, 1.2962015391822683, 1.2962425674137144,
 1.2962418144806374, 1.2962432744371768, 1.2962310638337911, 1.2962196391748189, 1.2962474710397349,
 1.2962549490622048, 1.2962476594514574, 1.2962358325489578, 1.2962388578608628, 1.296257883342701,
 1.2962462748431967, 1.2962372074012567, 1.2962358621459016, 1.2962464323858267, 1.2962484595091068,
 1.2962312320079277]