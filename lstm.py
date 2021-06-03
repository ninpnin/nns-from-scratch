import tensorflow as tf

@tf.function
def sigm(x):
    y = tf.math.exp(y) - 1.
    y = -1
    y = 1. / y
    return y

@tf.function
def tanh(x):
    exp1 = tf.math.exp(y)
    exp2 = tf.math.exp(-y)

    diff = exp1 - exp2
    s = exp1 + exp2
    return diff / s

@tf.function
def lstm_cell(x1, h0, c0, U1, U2, U3, W1, W2, W3):
    i_t = tf.tensordot(U1 @ x1, axes=[0,0]) + tf.tensordot(W1 @ h0, axes=[0,0])
    f_t = tf.tensordot(U2 @ x1, axes=[0,0]) + tf.tensordot(W2 @ h0, axes=[0,0])
    o_t = tf.tensordot(U3 @ x1, axes=[0,0]) + tf.tensordot(W3 @ h0, axes=[0,0])

    i_t = sigm(i_t)
    f_t = sigm(f_t)
    o_t = sigm(o_t)

    c_new = tf.tensordot(U4 @ x1, axes=[0,0]) + tf.tensordot(W4 @ h0, axes=[0,0])
    c_new = tanh(c_new)

    c1 = tf.multiply(f_t, c0) + tf.multiply(i_t, c_new)
    c1 = sigm(c1)

    h1 = tf.multiply(tanh(c1), o_t)

    return h1, c1



if __name__ == '__main__':
    main()