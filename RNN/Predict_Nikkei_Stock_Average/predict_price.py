import keras
import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler

lookback = 30

def get_data():
    data_file = 'nikkei_stock_average_daily_jp.csv'
    df = pd.read_csv(data_file, index_col=0, encoding='cp932', skipfooter=1, engine='python')
    closing_price = df[['終値']].values

    closing_price = closing_price[-lookback:]
    latest_date = df.index.values[-1]
    return closing_price, latest_date


def download_stock_data():
    url = 'https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_daily_jp.csv'
    try:
        urllib.request.urlretrieve(url,'nikkei_stock_average_daily_jp.csv')
    except:
        print('ファイルをダウンロードできませんでした。')


def main():
    download_stock_data()

    closing_price, latest_date = get_data()

    scaler = StandardScaler()
    scaler.fit(closing_price)
    closing_price_std = scaler.transform(closing_price).reshape(-1, lookback, 1)

    model = keras.models.load_model('model.h5', compile=False)

    df_predict_std = pd.DataFrame(model.predict(closing_price_std), columns=['予測値'])
    predict = scaler.inverse_transform(df_predict_std['予測値'].values)

    print('\n-------------------------------------------------------------------------------')
    print('      ('+latest_date+') Predict the next closing price(Nikkei 225): ' + str(predict[0]))
    print('-------------------------------------------------------------------------------\n')


if __name__ =='__main__':
    main()
