# 回帰問題,損失関数: 平均二乗誤差（MSE: Mean Squared Error）
# 回帰問題,評価指標: 平均絶対誤差（MAE: Mean Absolute Error）

import pathlib
import pandas as pd
# import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# 正規化
def _norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']


def load_data():
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)
    return raw_dataset

def cleansing(raw_dataset):
    # クレンジング
    dataset = raw_dataset.copy()
    print(dataset.tail())

    dataset.isna().sum()
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0
    print(dataset.tail())

    # 分割
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    # 目的変数の分離
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    normed_train_data = _norm(train_dataset, train_stats)
    normed_test_data = _norm(test_dataset, train_stats)
    return (normed_train_data, train_labels), (normed_test_data, test_labels)


def build_model(normed_train_data):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(normed_train_data.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()
    return model


# エポックが終わるごとにドットを一つ出力することで進捗を表示
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


def train_model(model, normed_train_data, train_labels):
    epochs = 200
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(normed_train_data, train_labels, epochs=epochs,
    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    return history


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig, ax = plt.subplots(2, 1, figsize=(7,12))
    ax = ax.ravel()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Mean Abs Error [MPG]')
    ax[0].plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    ax[0].plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    ax[0].legend()
    ax[0].set_ylim([0, 5])

    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Mean Square Error [$MPG^2$]')
    ax[1].plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    ax[1].plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    ax[1].legend()
    ax[1].set_ylim([0, 20])


def show_prediction_difference(model, normed_test_data, test_labels):
    test_predictions = model.predict(normed_test_data).flatten()
    fig, ax = plt.subplots(1, 1, figsize=(7,10))
    ax.scatter(test_labels, test_predictions)
    ax.set_xlabel('True Values [MPG]')
    ax.set_ylabel('Predictions [MPG]')
    ax.axis('square')
    ax.set_ylim([0, 50])
    ax.set_xlim([0, 50])
    ax.plot([-100, 100], [-100, 100])
    plt.show()

def main():
    raw_dataset = load_data()
    (normed_train_data, train_labels), (normed_test_data, test_labels) = cleansing(raw_dataset)
    model = build_model(normed_train_data)
    history = train_model(model, normed_train_data, train_labels)
    plot_history(history)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
    print("Testing set Mean Absolute Error: {:5.2f} MPG".format(mae))

    # 予測と結果を図示
    show_prediction_difference(model, normed_test_data, test_labels)


if __name__ == '__main__':
    main()
