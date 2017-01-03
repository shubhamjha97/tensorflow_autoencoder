import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist=input_data.read_data_sets('/tmp/data/', one_hot=True)

n_input=28*28
n_hidden=400
n_output=28*28
batch_size=100
ex_to_show=10
no_epochs=1

X=tf.placeholder('float', [None, n_input])
weights={'w1':tf.Variable(tf.random_normal([n_input, n_hidden])), 'w2':tf.Variable(tf.random_normal([n_hidden, n_output]))}
biases={'b1':tf.Variable(tf.random_normal([n_input])), 'b2':tf.Variable(tf.random_normal([n_hidden])), 
		'b3':tf.Variable(tf.random_normal([n_output]))}
layer_h1=tf.nn.sigmoid(tf.add(tf.matmul(X, weights['w1']), biases['b2']))
output=tf.nn.sigmoid(tf.add(tf.matmul(layer_h1, weights['w2']),biases['b3']))

y_true=X

cost=tf.reduce_mean(tf.pow((output-y_true), 2))
optimizer=tf.train.RMSPropOptimizer(0.1).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	no_batch=int(mnist.train.num_examples/batch_size)

	for epoch in range(no_epochs):
		for current_batch in range(no_batch):
			batch_x, batch_y=mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={X:batch_x})
		print 'Epoch:' + str(epoch+1) + ' Cost:'+ str(c)

	print 'Training Complete'
	#test set
	decoded, hidden=sess.run([output, layer_h1], feed_dict={X:mnist.test.images[:ex_to_show]})
	f,a = plt.subplots(3,10, figsize=(10,3))
	for i in range(ex_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='gray')
        a[1][i].imshow(np.reshape(decoded[i], (28, 28)), cmap='gray')
        a[2][i].imshow(np.reshape(hidden[i], (20,20)), cmap='gray')
   	f.show()
   	#plt.show()
   	print decoded
   	plt.waitforbuttonpress()