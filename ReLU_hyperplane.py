import numpy as np
from matplotlib import pyplot as plt

PI=np.pi
PI2=2*PI
def gencircle(rc,rr=0.1,offset=[0,0],num=100):
    c=[]
    for i in range(num):
        r=rc+np.random.uniform(-rr,rr,1)
        th=np.random.uniform(0,PI2,1)
        c.append([r*np.sin(th)+offset[0],r*np.cos(th)+offset[1]])
    return np.c_[np.array(c).reshape(num,2)]

num = 200
plt.figure(figsize=(5,5))

circles=np.vstack([gencircle(0.7,0.12,num=num)])
plt.scatter(circles[:, 0], circles[:, 1])

x = np.random.normal(0, 0.12, num)
y = np.random.normal(0, 0.12, num)
X = np.vstack((x, y)).T

#, np.repeat(1, num)

plt.scatter(X[:, 0], X[:, 1])
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()


from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([

])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

X_train = np.vstack([X, circles])
y_train = np.hstack([np.repeat(0,num), np.repeat(1,num)])


X_train = X_train.reshape(num*2, 2)
y_train = y_train.reshape(num*2, 1)
print(X_train.shape)
print(y_train.shape)

model.fit(X_train, y_train, epochs=100, batch_size=32)
