import tensorflow as tf
import numpy as np
import math

###### models ######
class CNN:
  ## todo - implement predict function
  def __init__(self, params):
    """params has width, depth, numParam"""
    self.params = params
    self.network_name = 'CNN'
    self.sess = tf.Session()

    self.In = tf.placeholder('float', [None, params['width'], params['height'], 4],
                            name='In')
    self.Labels = tf.placeholder('float', [None, params['numParam']],name='Labels')

    # Layer 1 (Convolutional)
    layer_name = 'conv1'
    size = 10
    channels = 4
    filters = 32
    stride = 5
    self.w1 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
                          name=self.network_name + '_' + layer_name + '_weights')
    self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
    self.c1 = tf.nn.conv2d(self.In, self.w1, strides=[1, stride, stride, 1], padding='SAME',
                           name=self.network_name + '_' + layer_name + '_convs')
    self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), name=self.network_name + '_' + layer_name + '_activations')

    # Layer 2 (Convolutional)
    layer_name = 'conv2'
    size = 10
    channels = 32
    filters = 64
    stride = 5
    self.w2 = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01),
                          name=self.network_name + '_' + layer_name + '_weights')
    self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
    self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',
                           name=self.network_name + '_' + layer_name + '_convs')
    self.o2 = tf.nn.relu(tf.add(self.c2, self.b2), name=self.network_name + '_' + layer_name + '_activations')

    o2_shape = self.o2.get_shape().as_list()

    # Layer 3 (Fully connected)
    layer_name = 'fc3'
    hiddens = 256
    dim = o2_shape[1] * o2_shape[2] * o2_shape[3]
    self.o2_flat = tf.reshape(self.o2, [-1, dim], name=self.network_name + '_' + layer_name + '_input_flat')
    self.w3 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
                          name=self.network_name + '_' + layer_name + '_weights')
    self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
    self.ip3 = tf.add(tf.matmul(self.o2_flat, self.w3), self.b3, name=self.network_name + '_' + layer_name + '_ips')
    self.o3 = tf.nn.relu(self.ip3, name=self.network_name + '_' + layer_name + '_activations')

    # Layer 4 (Fully connected output)
    layer_name = 'fc4'
    hiddens = params['numParam']
    dim = 256
    self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
                          name=self.network_name + '_' + layer_name + '_weights')
    self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name + '_' + layer_name + '_biases')
    self.logits = tf.add(tf.matmul(self.o3, self.w4), self.b4, name='logits')
    self.Out = tf.nn.sigmoid(self.logits, name='Out')

    # Cost,Optimizer
    #self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.Out, self.Labels), 2))
    self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Labels)

    if self.params['load_file'] is not None:
      self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step', trainable=False)
    else:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Gradient descent on loss function
    self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'], epsilon=self.params['rms_eps']).minimize(self.cost,
                                                                                                         global_step=self.global_step)

    self.saver = tf.train.Saver(max_to_keep=0)

    self.sess.run(tf.global_variables_initializer())

    if self.params['load_file'] is not None:
      print('Loading checkpoint...')
      self.saver.restore(self.sess, self.params['load_file'])

  def train(self, bat_In, labels):
    feed_dict = {self.In: bat_In, self.Labels: labels}
    _, cnt, cost = self.sess.run([self.rmsprop, self.global_step, self.cost], feed_dict=feed_dict)
    return cnt, cost/bat_In.shape[0]

  def predict(self,bat_In):
    feed_dict = {self.In: bat_In}
    out = self.sess.run([self.Out], feed_dict=feed_dict)
    prediction = np.round(out)
    return prediction

  def save_ckpt(self, filename):
    self.saver.save(self.sess, filename)

###### helper functions ######
def upsample_layer(bottom,
               n_channels, name, upscale_factor):

  kernel_size = 2 * upscale_factor - upscale_factor % 2
  stride = upscale_factor
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    # Shape of the bottom tensor
    in_shape = tf.shape(bottom)

    h = ((in_shape[1] - 1) * stride) + 1
    w = ((in_shape[2] - 1) * stride) + 1
    new_shape = [in_shape[0], h, w, n_channels]
    output_shape = tf.stack(new_shape)

    filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

    weights = get_bilinear_filter(filter_shape, upscale_factor)
    deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                    strides=strides, padding='SAME')

  return deconv