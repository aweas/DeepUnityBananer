import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np


class QNetworkTf():
    """Actor (Policy) Model."""

    def __init__(self, session, state_size, action_size, name, checkpoint_file=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.sess = session
        self.name = name

        if checkpoint_file is None:
            with tf.variable_scope("placeholders_"+self.name):
                self.input = tf.placeholder(tf.float32, shape=(None, state_size), name='input')
                self.y_input = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')
                self.gather_index = tf.placeholder(tf.int32, shape=(None), name='gather_index')

            self.output = self._inference()
            self.loss, self.optimizer = self._training_graph()

            self.sess.run([tf.global_variables_initializer(),
                           tf.local_variables_initializer()])
        else:
            checkpoint_dir = '/'.join(checkpoint_file.split('/')[:-1])
            saver = tf.train.import_meta_graph(checkpoint_file+'.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))

            self.input = tf.get_default_graph().get_tensor_by_name('placeholders_'+self.name+'/input:0')
            self.y_input = tf.get_default_graph().get_tensor_by_name('placeholders_'+self.name+'/y_input:0')
            self.gather_index = tf.get_default_graph().get_tensor_by_name('placeholders_'+self.name+'/gather_index:0')
            self.loss = tf.get_default_graph().get_tensor_by_name(f'training_{self.name}/loss:0')
            self.optimizer = tf.get_default_graph().get_operation_by_name(f'training_{self.name}/optimize')
            self.output = tf.get_default_graph().get_tensor_by_name(f'inference_{self.name}/dense_2/BiasAdd:0')

        self.step = 0

    def _inference(self):
        with tf.variable_scope("inference_"+self.name):
            layer = tf.layers.dense(self.input, 128, activation=tf.nn.relu)
            layer = tf.layers.dense(layer, 128, activation=tf.nn.relu)
            output = tf.layers.dense(layer, 4)
        return layer

    def _training_graph(self):
        with tf.variable_scope('training_'+self.name):
            pad = tf.range(tf.size(self.gather_index))
            pad = tf.expand_dims(pad, 1)
            ind = tf.concat([pad, self.gather_index], axis=1)

            gathered = tf.gather_nd(self.output, ind)
            gathered = tf.expand_dims(gathered, 1)
            loss = tf.losses.mean_squared_error(
                labels=self.y_input, predictions=gathered)
            # loss = tf.multiply(self.loss_modifier, loss)
            loss = tf.reduce_mean(loss, name='loss')

            optimize = tf.train.AdamOptimizer(
                learning_rate=5e-4).minimize(loss, name='optimize')

        return loss, optimize

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.sess.run(self.output, feed_dict={self.input: state})

    def train(self, states, y_correct, actions):
        reduced, result, _ = self.sess.run([self.loss, self.output, self.optimizer], feed_dict={
            self.input: states, self.y_input: y_correct, self.gather_index: actions})
        return reduced, result
