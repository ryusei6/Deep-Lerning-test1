import pandas as pd
import stock_price_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

lookback = 30
def get_data():
    data_file = 'nikkei_stock_average_daily_jp.csv'
    df = pd.read_csv(data_file, index_col=0, encoding='cp932', skipfooter=1, engine='python')
    closing_price = df[['終値']].values

    index = -lookback
    X_train = closing_price[index:]
    return X_train

def main():
    X_train = get_data()

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train).reshape(-1, lookback, 1)

    model = stock_price_model.load_model()

    df_predict_std = pd.DataFrame(model.predict(X_train_std), columns=['予測値'])
    predict = scaler.inverse_transform(df_predict_std['予測値'].values)
    
    print('------------------------------------------')
    print('predict price: ' + str(predict[0]))
    print('------------------------------------------')

if __name__ =='__main__':
    main()
