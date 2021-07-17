import tensorflow as tf

@tf.function
def sigm(x):
    y = tf.math.exp(x) - 1.
    #y = -1 * y
    y = 1. / y
    return y

@tf.function
def tanh(x):
    exp1 = tf.math.exp(x)
    exp2 = tf.math.exp(-x)

    diff = exp1 - exp2
    s = exp1 + exp2
    return diff / s

@tf.function
def lstm_cell(x1, h0, c0, weights):
    U1, U2, U3, U4, W1, W2, W3, W4 = weights
    i_t = tf.tensordot(U1, x1, axes=[0,0]) + tf.tensordot(W1, h0, axes=[0,0])
    f_t = tf.tensordot(U2, x1, axes=[0,0]) + tf.tensordot(W2, h0, axes=[0,0])
    o_t = tf.tensordot(U3, x1, axes=[0,0]) + tf.tensordot(W3, h0, axes=[0,0])

    i_t = sigm(i_t)
    f_t = sigm(f_t)
    o_t = sigm(o_t)

    c_new = tf.tensordot(U4, x1, axes=[0,0]) + tf.tensordot(W4, h0, axes=[0,0])
    c_new = tanh(c_new)

    c1 = tf.multiply(f_t, c0) + tf.multiply(i_t, c_new)
    c1 = sigm(c1)

    h1 = tf.multiply(tanh(c1), o_t)

    return h1, c1

def initialize_weights(hidden_size = 8):
    U1 = tf.random.normal([hidden_size, hidden_size])
    U2 = tf.random.normal([hidden_size, hidden_size])
    U3 = tf.random.normal([hidden_size, hidden_size])
    U4 = tf.random.normal([hidden_size, hidden_size])

    W1 = tf.random.normal([hidden_size, hidden_size])
    W2 = tf.random.normal([hidden_size, hidden_size])
    W3 = tf.random.normal([hidden_size, hidden_size])
    W4 = tf.random.normal([hidden_size, hidden_size])

    return U1, U2, U3, U4, W1, W2, W3, W4

def initialize_values(hidden_size = 8):
    x1 = tf.random.normal([hidden_size])
    h0 = tf.random.normal([hidden_size])
    c0 = tf.random.normal([hidden_size])

    return x1, h0, c0

def main():
    weights = initialize_weights()
    x1, h0, c0 = initialize_values()

    print(x1)
    h1, c1 = lstm_cell(x1, h0, c0, weights)

    print(h1)

if __name__ == '__main__':
    main()