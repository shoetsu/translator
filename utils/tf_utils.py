#coding: utf-8
import tensorflow as tf

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]
