import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation

maxlen = 47

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

# sin波にノイズを付与する
def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

f = toy_problem()

#データセット定義
def make_dataset(low_data):
    data,target=[],[]
    # maxlen = 25

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i+maxlen])
        target.append(low_data[i+maxlen])

    re_data = np.array(data).reshape(len(data),maxlen,1)
    re_target = np.array(target).reshape(len(data),1)

    return re_data,re_target

x_train, y_train = make_dataset(f)


#モデル
length_of_sequence = x_train.shape[1]
n_hidden = 512

model = Sequential()
model.add(LSTM(n_hidden,batch_input_shape=(None,length_of_sequence,1),return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


# 学習
early_stopping = EarlyStopping(monitor='val_loss',mode='auto',patience=5)
model.fit(x_train, y_train,
         batch_size=256,
         epochs=100,
         callbacks=[early_stopping],
         validation_split=0.1)

# 表示
predicted = model.predict(x_train)
plt.figure()
plt.plot(range(0,len(f)),f,color='b',label='raw')
plt.plot(range(maxlen,len(predicted)+maxlen),predicted,color='r',label='predict')
plt.legend()
plt.show()



# 1つの学習データの時間の長さ -> 25
f = toy_problem(T=100)
x_train, y_train = make_dataset(f)
print(x_train.shape)
future_test = x_train[200-maxlen]
time_length = future_test.shape[0]

# 未来の予測データを保存していく変数
future_result = np.array([])

# 未来予想
array = []
for i in range(400):
    test_data= np.reshape(future_test, (1, time_length, 1))
    batch_predict = model.predict(test_data)
#     print(batch_predict.shape)
    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)

    future_result = np.append(future_result, batch_predict)

# sin波をプロット
plt.figure()
plt.plot(range(maxlen,len(predicted)+maxlen),predicted, color="r", label="predict_data")
plt.plot(range(0, len(f)), f, color="b", label="row_data")
plt.plot(range(0+len(f), len(future_result)+len(f)), future_result, color="g", label="future_predict")
plt.legend()
plt.show()
