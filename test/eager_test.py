import tensorflow as tf

tf.enable_eager_execution()
# tf.executing_eagerly()

x = [[2]]
m = tf.matmul(x, x)
print('m result {}'.format(m))