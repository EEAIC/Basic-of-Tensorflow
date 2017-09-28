import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

X = [1.0, 2.0, 3.0]
Y = [1, 2, 3]

W = tf.Variable(5.0)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# train = optimizer.minimize(cost)

gvs = optimizer.compute_gradients(cost)
train = optimizer.apply_gradients(gvs)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

