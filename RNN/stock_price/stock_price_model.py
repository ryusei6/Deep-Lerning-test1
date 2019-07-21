import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data_file = 'nikkei_stock_average_daily_jp.csv'

## 予測に必要な日数
lookback = 30

epochs = 200

df = pd.read_csv(data_file, index_col=0, encoding='cp932', skipfooter=1, engine='python')
closing_price = df[['終値']].values


def make_dataset(low_data):
    data, target = [], []
    for i in range(len(low_data)-lookback):
        data.append(low_data[i:i + lookback])
        target.append(low_data[i + lookback])
    re_data = np.array(data).reshape(len(data), lookback)
    re_target = np.array(target).reshape(len(data),1)
    return re_data, re_target


## 標準化
def standardization(X_train, y_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train).reshape(-1, lookback, 1)
    X_test_std = scaler.transform(X_test).reshape(-1, lookback, 1)
    scaler.fit(y_train)
    y_train_std = scaler.transform(y_train)
    return scaler, X_train_std, X_test_std, y_train_std


def create_model(X_train_std):
    model = Sequential()
    model.add(GRU(128, input_shape=(None, X_train_std.shape[-1]),return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    model.summary()
    return model


def show_loss(result):
    ## 訓練の損失値をプロット
    epochs = range(len(result.history['loss']))
    plt.plot(epochs, result.history['loss'], 'b', alpha=0.6, label='train', linewidth=1)
    plt.plot(epochs, result.history['val_loss'], 'r', alpha=0.6, label='test', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def show_predict(model, scaler, X_test_std, y_test):
    ## 予測値
    df_predict_std = pd.DataFrame(model.predict(X_test_std), columns=['予測値'])

    ## 予測値を元に戻す
    predict = scaler.inverse_transform(df_predict_std['予測値'].values)

    ## 予測結果をプロット
    pre_date = df.index[-len(y_test):].values
    plt.plot(pre_date, y_test, 'b', alpha=0.6, marker='.', label='raw', linewidth=1)
    plt.plot(pre_date, predict, 'r', alpha=0.6, marker='.', label='predict', linewidth=1)
    ticks = 30
    plt.xticks(range(0, len(pre_date), ticks), pre_date[::ticks])
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()


def load_model():
    model = keras.models.load_model('model.h5', compile=False)
    return model

def main():
    # 訓練、テスト用データ
    test_size = 0.2
    train_size = int(len(closing_price) * (1 - test_size))
    X_train, y_train = make_dataset(closing_price[0:train_size])
    X_test, y_test = make_dataset(closing_price[train_size:])

    scaler, X_train_std, X_test_std, y_train_std = standardization(X_train, y_train, X_test)

    # 学習済みモデル
    model = load_model()

    # 学習させる場合は下をコメントアウト
    # model = create_model(X_train_std)
    # result = model.fit(X_train_std, y_train_std,
    #                   batch_size=256,
    #                   epochs=epochs,
    #                   validation_split=0.1)

    # show_loss(result)

    show_predict(model, scaler, X_test_std, y_test)
    # model.save('model.h5', include_optimizer=False)


if __name__ == '__main__':
    main()
