import os, os.path
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim
import tensorflow.contrib.losses as L
import tensorflow.contrib.keras as K
import numpy as np 

class CNNRegressor():
    def __init__(self, configs, log_dir=None):
        self.parse_cfgs(configs)
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.h, self.w])
            self.y = tf.placeholder(tf.float32, [None,1])
            self.lr = tf.placeholder(tf.float32, [])    # learning rate
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')

            # initializer
            self.initializer = tf.contrib.layers.xavier_initializer 
            self.transfer = tf.nn.relu 
            
            self.hidden = self.build_embedder(self.x, self.n_filters, self.kernel_sizes, self.strides, 'convolutional')
            self.preds = self.build_regressor(self.hidden, self.hidden_sizes, self.output_dim, scope='regressor')

            self.mse = tf.reduce_mean(tf.square(self.preds - self.y))

            tf.contrib.losses.add_loss(self.mse)
            self.total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_opt = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

            init = tf.global_variables_initializer()

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=sess_config)
            self.sess.run(init)
            self.prepare_logger(log_dir)
        return None


    def parse_cfgs(self, cfgs):
        self.h = cfgs['h']
        self.w = cfgs['w']
        self.output_dim = 1
        self.n_filters = cfgs['n_filters']
        self.kernel_sizes = cfgs['kernel_sizes']
        self.strides = cfgs['strides']
        self.hidden_sizes = cfgs['hidden_sizes']
        self.keep_prob = cfgs['keep_prob']
        self.weight_decay = cfgs['weight_decay']
        self.use_bn = cfgs['use_bn']
        return None

    def prepare_logger(self, log_dir):
        # summary writer
        self.saver = tf.train.Saver(max_to_keep=10)
        if log_dir:
            self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("mse", self.mse)
            tf.summary.histogram("preds", self.preds)
            self.merged_summaries = tf.summary.merge_all()

    def build_embedder(self, inp, n_filters, kernel_sizes, strides, scope):
        # inp: shape: bs x t x n_freq_bins
        with tf.variable_scope(scope):
            output = tf.expand_dims(inp, -1) # shape: bs x t x n_freq_bins x 1
            for i in range(len(n_filters)):
                output = K.layers.Convolution2D(filters=n_filters[i], 
                    kernel_size=kernel_sizes[i], strides=strides[i])(output)
                if self.use_bn:
                    output = self.bn_layer(output, scope='bn_%d'%i)
                output = self.transfer(output)
                # pooling
                output = K.layers.MaxPooling2D(pool_size=[2, 1])(output)
            dim = output.shape[1] * output.shape[2] * output.shape[3]
            output = tf.reshape(output, [-1, dim.value])
        return output

    def build_regressor(self, inp, hidden_sizes, outdim, scope):
        out = inp
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected], 
                weights_initializer=self.initializer(),
                biases_initializer=tf.constant_initializer(0),
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(self.weight_decay)):              
                for i in range(len(hidden_sizes)):
                    out = slim.fully_connected(out, hidden_sizes[i], scope='fc%d'%(i+1))
                    if self.use_bn: 
                        out = self.bn_layer(out, scope='bn%d'%i)
                    out = self.transfer(out)
                    if self.keep_prob > 0 and self.keep_prob < 1:
                        out = slim.dropout(out, self.keep_prob, 
                            is_training=self.is_training, scope='dropout%d'%i)
                out = slim.fully_connected(out, outdim, scope='fc%d'%(len(hidden_sizes)+1))
                if self.use_bn: 
                    out = self.bn_layer(out, scope='bn%d'%(len(hidden_sizes)+1))
        return out


    def partial_fit(self, x, y, lr, get_summary=False):
        summary = None
        step = self.sess.run(self.global_step) 
        if get_summary:
            loss, preds, train_opt, summary = self.sess.run([self.total_loss, self.preds, 
                self.train_opt, self.merged_summaries], 
                feed_dict={self.x:x, self.y:y, self.lr:lr, self.is_training:True})
        else:
            loss, preds, train_opt = self.sess.run([self.total_loss, self.preds, self.train_opt], 
                feed_dict={self.x:x, self.y:y, self.lr:lr, self.is_training:True})
        return loss, preds, summary, step

    def calc_loss(self, x, y):
        loss = self.sess.run(self.total_loss, 
            feed_dict={self.x:x, self.y:y, self.lr:0, self.is_training:False})
        return loss

    def bn_layer(self, inputs, scope):
        bn = tf.contrib.layers.batch_norm(inputs, is_training=self.is_training, 
            center=True, fused=False, scale=True, updates_collections=None, decay=0.9, scope=scope)
        return bn

    def predict(self, x):
        dummy_y = np.zeros((x.shape[0], self.output_dim))
        preds = self.sess.run(self.preds, feed_dict={self.x:x, self.y:dummy_y, self.lr:0, self.is_training:False})
        return preds 

    def save(self, save_path, step):
        self.saver.save(self.sess, save_path, global_step=step)

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def log(self, summary):
        self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))

