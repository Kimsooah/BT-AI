import tensorflow as tf


def main():
    # define the graph
    a = tf.constant(1.0)
    b = tf.constant(1.0)
    c = tf.constant(4.0)
    d = tf.div(tf.add(a, b), c)

    # execute the graph
    with tf.Session() as session:
        print(session.run(d))  # 0.5


if __name__ == '__main__':
    print("tensorflow version : " + tf.__version__)
    main()