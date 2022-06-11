import tensorflow as tf
import numpy as np

class BilinearCNN2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BilinearCNN2D, self).__init__(**kwargs)

    # def build(self, input_shape):
        # print('build')
        # output_dim = input_shape[-1]
        # self.kernel = self.add_weight(
        #     shape=(output_dim * 2, output_dim),
        #     initializer=self.initializer,
        #     name="kernel",
        #     trainable=True,
        # )

    def call(self, inputs):
        # print('call')
        assert len(inputs) == 2
        left = inputs[0]
        # print('left:', left)
        right = inputs[1]
        # print('right:', right)
        l_shape = left.shape.as_list()
        # print('l_shape:', l_shape)
        assert tuple(l_shape) == tuple(right.shape.as_list())
        inner_dim = l_shape[1] * l_shape[2]
        outer_dim = l_shape[3]
        # print('inner_dim:', inner_dim)
        # print('outer_dim:', outer_dim)
        output_shape = tf.TensorSpec((None, outer_dim, outer_dim))
        # print('output_shape', output_shape)
        left = tf.reshape(left, (-1, inner_dim, outer_dim), name='r1')
        # print('left 2:', left)
        right = tf.reshape(right, (-1, inner_dim, outer_dim), name='r2')
        # print('right 2:', right)
        both = tf.stack([left, right], axis=0)
        # print('both:', both)
        swapped = tf.transpose(both, [1, 0, 2, 3])
        # print('swapped:', swapped)
        dotted = tf.map_fn(fn=lambda t: tf.tensordot(t[0], t[1], axes=[0,0]), elems=swapped)
        # print('dotted:', dotted)
        flat = tf.reshape(dotted, (-1, outer_dim * outer_dim), name='r3')
        sqrted = tf.map_fn(fn=lambda t: tf.math.sign(t) * tf.math.sqrt(tf.math.abs(t) + 1e-9), elems=flat)
        normed = tf.map_fn(fn=lambda t: tf.math.l2_normalize(t, axis=-1), elems=sqrted)
        output = normed
        return output
    
""" Given a pair of Flattened tensors, multiply those tensors """
class BilinearCNN1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BilinearCNN1D, self).__init__(**kwargs)

    # def build(self, input_shape):
        # print('build')
        # output_dim = input_shape[-1]
        # self.kernel = self.add_weight(
        #     shape=(output_dim * 2, output_dim),
        #     initializer=self.initializer,
        #     name="kernel",
        #     trainable=True,
        # )
        
    def call(self, inputs):
        # print('call')
        assert len(inputs) == 2
        left = inputs[0]
        # print('left:', left)
        right = inputs[1]
        # print('right:', right)
        l_shape = left.shape.as_list()
        print('l_shape:', l_shape)
        assert len(tuple(l_shape)) == 2 
        assert tuple(l_shape) == tuple(right.shape.as_list())
        inner_dim = l_shape[1]
        outer_dim = l_shape[2]
        # print('inner_dim:', inner_dim)
        # print('outer_dim:', outer_dim)
        output_shape = tf.TensorSpec((None, outer_dim, outer_dim))
        # print('output_shape', output_shape)
        both = tf.stack([left, right], axis=0)
        # print('both:', both)
        swapped = tf.transpose(both, [1, 0, 2, 3])
        # print('swapped:', swapped)
        dotted = tf.map_fn(fn=lambda t: tf.tensordot(t[0], t[1], axes=[0,0]), elems=swapped)
        # print('dotted:', dotted)
        flat = tf.reshape(dotted, (-1, outer_dim * outer_dim), name='r3')
        sqrted = tf.map_fn(fn=lambda t: tf.math.sign(t) * tf.math.sqrt(tf.math.abs(t) + 1e-9), elems=flat)
        normed = tf.map_fn(fn=lambda t: tf.math.l2_normalize(t, axis=-1), elems=sqrted)
        output = normed
        return output
