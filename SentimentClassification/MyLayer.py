# -*- encoding:utf-8 -*-
__author__ = 'SUTL'

import theano
import theano.tensor as T


def get_att(X, index):
    """
    Input attention, single sentence.
    Args:
        X: tensor, shape=[n, embed_dim]
        index: int, target index
    Return:
        tensor, shape=[n, embed_dim]
    """
    result, update = theano.scan(lambda v, u: T.dot(v, T.transpose(u)), sequences=X, non_sequences=X[index])
    result_soft = T.nnet.softmax(result)
    A = T.diag(T.flatten(result_soft))  # n×n
    return T.dot(A, X)  # [n, embed_dim]


def get_input_att(Xs, target_indices):
    """
    Input attention (batch)
    Args:
        Xs: tensor, shape=[batch_size, n, embed_dim]
        target_indices: int vector, target index
    Return:
        results: tensor, shape=[batch_size, n, embed_dim]
    """
    results, updates = theano.scan(get_att, sequences=[Xs, target_indices])
    return results 


class AttPoolLayer(Layer):

    ID = 1

    def __init__(self, input_dim, output_dim, nb_classes, **kwargs):
        """
        Args:
            output_dim: int
            nb_classes: int, number of classes
        """
        # self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_classes = nb_classes
        super(AttPoolLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        U_value = np.random.normal(size=(input_shape[-1], self.output_dim)).astype('float32')
        self.U = theano.shared(U_value, name='param_U'+str(AttPoolLayer.ID))  # params
        WL_value = np.random.normal(size=(self.output_dim, self.nb_classes)).astype('float32')
        self.WL = theano.shared(WL_value, name='param_WL'+str(AttPoolLayer.ID))  # params
        AttPoolLayer.ID += 1
        self.trainable_weights = [self.U, self.WL]
        super(AttPoolLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        """
        Args:
            x: tensor, shape=[batch_size, n, nb_filter](若上一层是conv)
        Return:
            tensor, shape=[batch_size, nb_filter]
        """
        return self.att_based_pooling(x)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def pool_one(self, R):
        """
        Attention-based pooling
        Args:
            R: tensor, sentence representation, shape=[n, nb_filter]
        Return:
            W_max: shape=[class_embbed_dim,]
        """
        G = theano.dot(theano.dot(R, self.U), self.WL)  # [n, nb_classes]
        A = T.nnet.softmax(G.transpose()).transpose()  # [n, nb_classes]
        WO = T.dot(R.transpose(), A)  # [nb_filter, nb_classes]
        W_max = T.max(WO, axis=1)  # [nb_filter,]
        return T.tanh(W_max)

    def att_based_pooling(self, Rs):
        """
        Attention-based pooling
        Args:
            Rs: tensor, shape=[batch_size, n, nb_filter]
        Return:
            tensor, shape=[bs, nb_filter]
        """
        results, updates = theano.scan(self.pool_one, sequences=Rs)
        return results  # shape=[batch_size, nb_filter]


