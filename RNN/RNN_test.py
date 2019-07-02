from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt


def sin(x, T=100):
    return np.sin(2.0*np.pi*x/T)

def cos(x, T=100):
    return np.cos(2.0*np.pi*x/T)

def toy_problem(T=300):
    x = np.arange(0,2*T+1)
    noise = np.random.uniform(low=-0.3,high=0.3,size=len(x))
    return sin(x) + cos(x/2) + noise
f = toy_problem()
# plt.plot(f)


#データセット定義
def make_dataset(low_data):
    data,target=[],[]
    maxlen = 25

    for i in range(len(low_data)-maxlen):
        data.append(low_data[i:i+maxlen])
        target.append(low_data[i+maxlen])

    re_data = np.array(data).reshape(len(data),maxlen,1)
    re_target = np.array(target).reshape(len(data),1)

    return re_data,re_target

x_train, y_train = make_dataset(f)

print(x_train.shape)


#モデル
length_of_sequence = x_train.shape[1]
n_hidden = 512

model = Sequential()
model.add(GRU(n_hidden,batch_input_shape=(None,length_of_sequence,1),return_sequences=False))
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
plt.plot(range(25,len(predicted)+25),predicted,color='r',label='predict')
plt.legend()
plt.show()
