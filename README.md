# index-word-rnn
这个实验用来验证RNN是否有索引的功能，因为对话模型本身是将问题编码到一个特定的编码，然后约束回复的生成。其实就是一个索引的功能。
前面的三个实验是用来验证RNN具有生成语义通顺的句子的能力，这个实验是为了验证RNN具有索引的功能。

代码是在word-rnn-with-theano上的改进，代码的本地目录

实验方法：
还是使用word-rnn-with-theano同样的数据，这次把句子数目缩减到1512个，字典大小变为2089

Implements the RNN encoder-decoder framework from Cho et al.

每句话首先被编码成bag-of-words向量 e，作为生成这句话本身的上下文（索引）。

nc : size of dict  ( number of classes)
de : dimension of embedding
nh : dimension of the hidden layers
