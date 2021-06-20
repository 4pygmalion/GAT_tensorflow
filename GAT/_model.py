import tensorflow as tf


class GraphAttentionLayer(tf.keras.layers.Layer):
    '''
    Single head attention GAT
    Callable instance of class

    Parameters
    ----------
    input_dim: int. the number of feature space of given X
    output_dim: int. the number of expected feature space
    head_num: int. (defaualt 1)

    Output
    ------
    tf.Tensor (N, output_dim)
    '''
    def __init__(self, input_dim, output_dim, head_num=1):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_num = head_num

    def build(self, input_shape):
        '''

        Parameters
        ----------
        input: h = (N, F)
        output: h' = (N, F')
        Returns
        -------

        '''
        # parameterized by a weight matrix W (F', F)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.output_dim), name='W')  # W:(F, F')

        # Parameterized by a weight vector a (2F')
        self.a = self.add_weight(shape=(2*self.output_dim, 1), name='a')
        self.built = True

    def call(self, X):
        # Eqation 1) mapping F feature space to F' features space
        features_i = tf.einsum('ij,jk->ik', X, self.kernel)  # (NxF) (FxF') => WH_i (NxF')
        features_j = tf.einsum('ij,jk->ik', X, self.kernel)  # (NxF) (FxF') => WH_j (NxF')

        # Equation 3) Attention coefficient
        e_ij = tf.tensordot(tf.concat([features_i, features_j], axis=1), self.a, axes=1)
        a_ij = tf.nn.softmax(e_ij, axis=0)  # (N,1)

        # Equation 4) Applying non-linearity with sigma
        context_vec = tf.einsum('ij,ik->ik', a_ij, features_i)  # (N,1) (NxF')
        h = tf.nn.sigmoid(context_vec, name='attention_coefficient')

        return h
