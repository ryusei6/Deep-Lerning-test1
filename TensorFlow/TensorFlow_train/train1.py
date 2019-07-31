import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

x = tf.constant(1.0)

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
