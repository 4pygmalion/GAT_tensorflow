##### Graph Attention Network + Tensorflow 2.x implementation
Graph Attention Networks published in ICLR 2018
for single head attention layer.
https://arxiv.org/abs/1710.10903
<br>


##### How to use?
```python 3
import tensorflow as tf
from GAT import GraphAttentionLayer

X = tf.random.uniform(shape=(30, 10))
gat_layer = GraphAttentionLayer(input_dim=10, output_dim=20) 
gat_layer(X)  # (30, 20)
```


##### Requirement
- Tensorflow 2.x 

##### How to install?
```bash
$ git clone https://github.com/4pygmalion/GAT_tensorflow.git
```

##### Core operation

```python 3
# Eqation 1) mapping F feature space to F' features space
features_i = tf.einsum('ij,jk->ik', X, self.kernel)  # (NxF) (FxF') => WH_i (NxF')
features_j = tf.einsum('ij,jk->ik', X, self.kernel)  # (NxF) (FxF') => WH_j (NxF')

# Equation 3) Attention coefficient
e_ij = tf.tensordot(tf.concat([features_i, features_j], axis=1), self.a, axes=1)
a_ij = tf.nn.softmax(e_ij, axis=0)  # (N,1)

# Equation 4) applying non-linearity with sigma
context_vec =  tf.einsum('ij,ik->ik', a_ij, features_i)  # (N,1) (NxF')
h = tf.nn.sigmoid(context_vec, name='attention_coefficient')
```