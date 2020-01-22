import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(np.nan_to_num(mask * mse)))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


# def mape(y_true, y_pred):
#     temp = np.abs((y_pred - y_true) / y_true)
#     return np.mean(temp)
#
#
# def rmse(y_true, y_pred):
#     temp = (y_true - y_pred) ** 2
#     return np.sqrt(np.mean(temp))
#
#
# def mae(y_true, y_pred):
#     temp = np.abs(y_true - y_pred)
#     return np.mean(temp)


# 标准化
# ss_x = StandardScaler()
# x = ss_x.fit_transform(x)
# ss_y = StandardScaler()
# y = ss_y.fit_transform(y.reshape(-1, 1)).squeeze()


# 切割数据集
def get_data(data, step, point):
    x = np.zeros((step, data.shape[0] - step))
    y = np.zeros((step, data.shape[0] - step))
    for i in range(data.shape[0] - step * 2):
        x[:, i] = data[i: i + step, point]
        y[:, i] = data[i + step: i + step * 2, point]
    x = x.transpose()
    y = y.transpose()
    # print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


# svr，预测12个时间片
def predict(x_train, x_test, y_train, pred_time):
    svr = SVR(kernel='linear', C=0.1)
    svr.fit(x_train, y_train[:, 0])
    y = svr.predict(x_test).reshape(-1, 1)
    for i in range(11):
        x_test = np.delete(x_test, 0, axis=1)
        x_test = np.append(x_test, y[:, -1].reshape(-1, 1), axis=1)
        y = np.append(y, svr.predict(x_test).reshape(-1, 1), axis=1)
    return y


data = np.load('data/PEMS03/PEMS03.npz')['data']
data = data.squeeze()

# data = np.load('data/PEMS07/PEMS07.npz')['data']
# data = data.squeeze()
# data = data[:17568]

print(data.shape)

step = 12
pred_time = 12

y_true3 = []
y_true2 = []
y_true1 = []
y_pred3 = []
y_pred2 = []
y_pred1 = []

# 每个观察点分别预测
for point in range(data.shape[1] - 2):
    x_train, x_test, y_train, y_test = get_data(data, step, point)
    y_true3.append(y_test)
    y_pred3.append(predict(x_train, x_test, y_train, pred_time))
    y_true2.append(y_true3[-1][:, :6])
    y_pred2.append(y_pred3[-1][:, :6])
    y_true1.append(y_true3[-1][:, :3])
    y_pred1.append(y_pred3[-1][:, :3])
    if point % 10 == 0:
        print(point)

y_pred3 = np.array(y_pred3).reshape(-1)
y_true3 = np.array(y_true3).reshape(-1)
y_pred2 = np.array(y_pred2).reshape(-1)
y_true2 = np.array(y_true2).reshape(-1)
y_pred1 = np.array(y_pred1).reshape(-1)
y_true1 = np.array(y_true1).reshape(-1)

print('1 hour')
print('mae \t\t rmse \t\t mape')
print(masked_mae_np(y_true3, y_pred3, 0), masked_rmse_np(y_true3, y_pred3, 0), masked_mape_np(y_true3, y_pred3, 0))
print('30 minutes')
print('mae \t\t rmse \t\t mape')
print(masked_mae_np(y_true2, y_pred2, 0), masked_rmse_np(y_true2, y_pred2, 0), masked_mape_np(y_true2, y_pred2, 0))
print('15 minutes')
print('mae \t\t rmse \t\t mape')
print(masked_mae_np(y_true1, y_pred1, 0), masked_rmse_np(y_true1, y_pred1, 0), masked_mape_np(y_true1, y_pred1, 0))
