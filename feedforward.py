import tensorflow as tf

@tf.function
def relu(x):
    return tf.math.maximum(0., x)

@tf.function
def softmax(x):
    ex = tf.math.exp(x)
    s = tf.reduce_sum(ex)
    return ex / s

@tf.function
def dense(x, W, b, activation=True):
    y = tf.tensordot(W, x, axes=[1,0]) + b
    if activation:
        return relu(y)
    else:
        return y

def neural_network(x0, Ws, bs):
    x = x0
    for W,b in zip(Ws[:-1], bs[:-1]):
        x = dense(x, W, b)

    x = dense(x, Ws[-1], bs[-1], activation=False)
    return softmax(x)

if __name__ == '__main__':
    sizes = [2,3,2]

    Ws = zip(sizes[1:], sizes[:-1])
    Ws = [tf.random.normal((i,j)) for i,j in Ws]
    Ws = [tf.Variable(W) for W in Ws]

    #print(Ws)
    
    bs = sizes[1:]
    bs = [tf.random.normal((i,)) for i in bs]
    bs = [tf.Variable(b) for b in bs]

    #print(bs)

    for i in range(10):
        x0 = tf.random.normal((sizes[0],))
        print(x0)

        y = neural_network(x0, Ws, bs)

        print(y)
