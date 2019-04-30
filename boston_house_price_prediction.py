import numpy as np
import pandas as pd
data_frame=pd.read_csv("boston_x_y_train.csv")
test_frame=pd.read_csv("boston_x_test.csv",header=None)
data_frame.insert(13,"ones",np.ones(379))
test_frame.columns=data_frame.columns[0:13]
test_frame.insert(13,"ones",np.ones(127))
print(data_frame.shape)
print(data_frame.columns)
print(data_frame.head())
data=np.array(data_frame)
# max_y=data[:,data.shape[1]-1].max()
test=np.array(test_frame)
def cost(points,m):
    total_cost = 0
    M = len(points)
    for i in range(M):
        total_cost+=(1/M)*((points[i,(points.shape[1]-1)]-(m*points[i,0:(points.shape[1]-1)]).sum())**2)
    return total_cost


def step_gradient(points, learning_rate, m):
    y_sum = points[:, (points.shape[1] - 1)].sum()
    for j in range(len(m)):
        slope = 0
        for i in range(points.shape[0]):
            y_pred = (points[i, 0:(points.shape[1] - 1)] * m).sum()
            slope += (y_pred-points[i, points.shape[1] - 1]) * points[i, j]

        slope *= 2 / (points.shape[0])
        temp_m = m[j]
        m[j] = temp_m - learning_rate * slope

    return m
def gd(points,learning_rate,num_iterations):
    m=np.zeros(points.shape[1]-1)
    print(len(m))
    print(type(m))
    print(points.shape)
    for i in range(num_iterations):
        m=step_gradient(points,learning_rate,m)
        if(i%100==0):
            print(i, "cost:", cost(points, m))
    return m
learning_rate=0.06
num_iterations=1000
m=gd(data,learning_rate,num_iterations)
print(m)
pred=((test*m)).sum(axis=1)
np.savetxt("pre]dictions_boston.csv",pred, delimiter=',',fmt="%.7F")
