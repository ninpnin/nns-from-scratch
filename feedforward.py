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
def dense(x, theta, activation=True):
    W, b = theta
    y = tf.tensordot(W, x, axes=[1,0]) + b
    if activation:
        return relu(y)
    else:
        return y

def neural_network(x0, theta):
    x = x0
    for theta_i in theta[:-1]:
        x = dense(x, theta_i)

    x = dense(x, theta[-1], activation=False)
    return softmax(x)

if __name__ == '__main__':
    # Layer widths, including input and output layers
    sizes = [2,3,2]

    # Initialize parameters
    Ws = zip(sizes[1:], sizes[:-1])
    Ws = [tf.random.normal((i,j)) for i,j in Ws]
    Ws = [tf.Variable(W) for W in Ws]
    
    bs = sizes[1:]
    bs = [tf.random.normal((i,)) for i in bs]
    bs = [tf.Variable(b) for b in bs]

    theta = list(zip(Ws, bs))

    # Run network for 10 randomly generated x's
    for i in range(10):
        x0 = tf.random.normal((sizes[0],))
        y = neural_network(x0, theta)
        print(x0)
        print(y)
