import tensorflow as tf
import numpy as np
import glob
import random
import utils

# lambda for convolution function
conv = lambda inp, filt, str, pad, id : tf.nn.conv2d(input   = inp,
													 filter  = filt,
													 strides = [1, str, str, 1],
													 padding = pad,
													 name    = id)

# lambda for deconvolution function
deconv = lambda inp, filt, st, pad, name, op_shape : tf.nn.conv2d_transpose(value = inp,
														   	                filter = filt,
													                        output_shape = op_shape,
													                        strides = [1, st, st, 1],
													                        padding = pad,
													                        name = name)

class denoise:
	def __init__(self, sess, length=256, k1=3, k2=5, k3=5, l1= 20,
		l2=20, l3=20, batch_size=10, epochs=100):
		self.LENGTH = length
		self.K1 = k1
		self.K2 = k2
		self.K3 = k3

		self.L1 = l1
		self.L2 = l2
		self.L3 = l3

		self.batch_size = batch_size
		self.epochs = epochs
		self.sess = sess
		self.create_model()

	def create_model(self):
		self.img_train  = tf.placeholder(tf.float32, shape=[None, self.LENGTH, self.LENGTH, 3], name='img_input')
		self.img_label  = tf.placeholder(tf.float32, shape=[None, self.LENGTH, self.LENGTH, 3], name="img_label")

		self.w1         = tf.Variable(tf.truncated_normal(
								shape=[self.K1, self.K1, 3, self.L1],
								stddev=0.1),
							name='w1')
		self.conv1      = tf.nn.relu(conv(self.img_train, self.w1, 1, 'VALID', 'conv1'))
		self.conv1_     = tf.contrib.layers.batch_norm(self.conv1)

		self.w2         = tf.Variable(tf.truncated_normal(
								shape=[self.K2, self.K2, self.L1, self.L2],
								stddev=0.1),
							name='w2')
		self.conv2      = tf.nn.relu(conv(self.conv1_, self.w2, 1, 'VALID', 'conv2'))
		self.conv2_     = tf.contrib.layers.batch_norm(self.conv2)

		self.w3         = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L2, self.L3],
								stddev=0.1),
							name='w3')
		self.conv3      = tf.nn.relu(conv(self.conv2_, self.w3, 1, 'VALID', 'conv3'))

		self.dw1        = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L2, self.L3],
								stddev=0.1),
							name='dw1')
		self.deconv1    = tf.nn.relu(deconv(self.conv3, self.dw1, 1, 'VALID', 'deconv1', tf.shape(self.conv2_)))

		self.dw2        = tf.Variable(tf.truncated_normal(
								shape=[self.K2, self.K2, self.L1, self.L2],
								stddev=0.1),
							name='dw2')
		self.deconv2    = tf.nn.relu(deconv(self.deconv1, self.dw2, 1, 'VALID', 'deconv2', tf.shape(self.conv1_)))

		self.dw3        = tf.Variable(tf.truncated_normal(
								shape=[self.K1, self.K1, 3, self.L1],
								stddev=0.1),
							name='dw3')
		self.deconv3    = tf.nn.relu(deconv(self.deconv2, self.dw3, 1, 'VALID', 'deconv3', tf.shape(self.img_label)))

		self.L1 = 1
		self.L2 = 2

		self.error      = tf.reduce_mean(tf.losses.absolute_difference(self.deconv3, self.img_label, weights=self.L1)+
							tf.losses.mean_squared_error(self.deconv3, self.img_label, weights=self.L2))

		self.optim      = tf.train.AdamOptimizer().minimize(self.error)

		self.saver = tf.train.Saver(max_to_keep=1)

		self.error_graph = tf.summary.scalar("Error", self.error)

	def train_model(self):
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

		step_count = 1
		for e in range(self.epochs):
			self.file_list = glob.glob('../datasets/urban_and_natural_images/raw/*.jpg')
			# self.file_list = random.shuffle(file_list)
			for i in range(len(self.file_list)/self.batch_size):
				batch_x, batch_y = utils.load(self.batch_size, self.file_list[i*self.batch_size:(i+1)*self.batch_size])
				
				_, er, graph_ = self.sess.run([self.optim, self.error, self.error_graph],
					feed_dict={self.img_train: batch_x, self.img_label: batch_y})

				print "Epochs {}/{} Error {}".format(e, self.epochs, er)

				self.writer.add_summary(graph_, step_count)

				if (e % 100 == 1):
					self.save(self.chkpt, step_count)
				
				step_count += 1