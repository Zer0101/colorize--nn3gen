from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ConvolutionBatchNormalizer(object):
    """
    Helper class that groups the normalization logic and variables.        

    Use:                                                                      
        ema = tf.train.ExponentialMovingAverage(decay=0.99999)                  
        bn = ConvolutionBatchNormalizer(depth, 0.001, ema, True)           
        update_assignments = bn.get_assigner()                                
        x = bn.normalize(y, train=training?)                                  
        (the output x will be batch-normalized).                              
    """

    def __init__(self, depth, epsilon, ema_trainer, scale_after_norm):
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
        self.variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
        self.ema_trainer = ema_trainer
        self.epsilon = epsilon
        self.scale_after_norm = scale_after_norm

    def get_assigner(self):
        """
            Returns an EMA apply op that must be invoked after optimization.
        """
        return self.ema_trainer.apply([self.mean, self.variance])

    def normalize(self, x, train=True):
        """
            Returns a batch-normalized version of x.
        """
        if train is not None:
            mean, variance = tf.nn.moments(x=x, axes=[0, 1, 2])
            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_norm_with_global_normalization(
                    x, mean, variance, self.beta, self.gamma,
                    self.epsilon, self.scale_after_norm)
        else:
            mean = self.ema_trainer.average(self.mean)
            variance = self.ema_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, local_beta, local_gamma,
                self.epsilon, self.scale_after_norm)


def batch_normalize(x, depth, phase_train):
    with tf.variable_scope('batch_normalize'):
        ema = tf.train.ExponentialMovingAverage(decay=0.99999)
        normalizer = ConvolutionBatchNormalizer(depth, 0.001, ema, True)
        normalizer.get_assigner()
        x = normalizer.normalize(x, train=phase_train)
    return x
