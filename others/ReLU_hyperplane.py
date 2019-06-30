import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import Dense, Activation

def _generate_r_circle(r=1,num=100):
    PI = np.pi
    theta = np.arange(0,360,360/num)
    x = r * np.cos(theta/180*PI)
    y = r * np.sin(theta/180*PI)
    x_noise = np.random.normal(0, 0.15, num)
    y_noise = np.random.normal(0, 0.15, num)
    x += x_noise
    y += y_noise

    r_circle_array = np.vstack((x,y)).T
    return r_circle_array


def _generate_c_circle(num=100):
    x = np.random.normal(0, 0.2, num)
    y = np.random.normal(0, 0.2, num)
    c_circle_array = np.vstack((x,y)).T # vstack:縦方向に連結
    return c_circle_array


def show_distribution(num=100):
    radius = 1.5

    r_circle = _generate_r_circle(r=radius, num=num)
    c_circle = _generate_c_circle(num=num)

    plt.figure(figsize=(5,5))
    plt.scatter(r_circle[:,0],r_circle[:,1])
    plt.scatter(c_circle[:,0],c_circle[:,1])
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    plt.show()
    return c_circle,r_circle


def create_model():
    model = Sequential([
        Dense(5, input_dim=2),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid'),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, c_circle, r_circle, num=100):
    x_train = np.vstack([c_circle, r_circle])
    y_train = np.hstack([np.repeat(0,num), np.repeat(1,num)])
    x_train = x_train.reshape(num*2, 2)
    y_train = y_train.reshape(num*2, 1)

    model.fit(x_train, y_train, epochs=100, batch_size=32)
    return x_train

def _plotModelOut(x,y,model):
    '''
    x,y: 2D MeshGrid input
    model: Keras Model API Object
    '''
    grid = np.stack((x,y))
    grid = grid.T.reshape(-1,2)
    outs = model.predict(grid)
    y1 = outs.T[0].reshape(x.shape[0],x.shape[0])
    contour = plt.contourf(x,y,y1, levels=[0.0, 0.3, 0.5, 0.7, 1.0], cmap='magma')
    plt.colorbar(contour)

def show_result(model, x_train):
    a = np.linspace(-2.5, 2.5, 100)
    xx, yy = np.meshgrid(a, a)

    plt.figure(figsize=(6,5))
    _plotModelOut(xx, yy, model)

    plt.scatter(x_train[200:, 0], x_train[200:, 1])
    plt.scatter(x_train[:200, 0], x_train[:200, 1])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.show()


def main():
    num = 200

    c_circle,r_circle = show_distribution(num=num)
    model = create_model()
    x_train = train_model(model, c_circle, r_circle, num=num)
    show_result(model, x_train)


if __name__ == '__main__':
    main()
