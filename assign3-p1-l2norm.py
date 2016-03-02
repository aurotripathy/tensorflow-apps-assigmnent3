# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = '/home/tempuser/PycharmProjects/tf-class-1/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28 # Squared
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

# The knobs and dials!
batch_size = 128
hidden_nodes = 1024 # Tried double the cells
starter_learn_rate = 0.5
reg_beta = 5e-4
num_steps = 200001

# Graph with stochastic gradient descent
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    keep_prob = tf.placeholder("float")

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes]))
    biases_1 = tf.Variable(tf.zeros([hidden_nodes]))

    # Training computation.
    h_logits = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)

    # Variables.
    weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(h_logits, weights_2) + biases_2
    logits = tf.nn.dropout(logits, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # L2 regularization for the fully connected parameters.
    # https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/mnist/convolutional.py
    regularizers = (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(biases_1) +
                    tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(biases_2))
    # Add the regularization term to the loss.
    loss += reg_beta * regularizers

    loss_summary = tf.scalar_summary("loss", loss)

    global_step = tf.Variable(0)  # count the number of steps taken.
    learn_rate = tf.train.exponential_decay(starter_learn_rate, global_step, 500, 0.96)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    valid_h_logits = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    valid_o_logits = tf.matmul(valid_h_logits, weights_2) + biases_2
    valid_prediction = tf.nn.softmax(valid_o_logits)

    test_h_logits = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
    test_o_logits = tf.matmul(test_h_logits, weights_2) + biases_2
    test_prediction = tf.nn.softmax(test_o_logits)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()


#let's run it
with tf.Session(graph=graph) as session:

    writer = tf.train.SummaryWriter("/tmp/notmnist_logs", session.graph_def)

    tf.initialize_all_variables().run()
    print("Initialized")

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.

        if step % 500 == 0:
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 1.0 }
            _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learn_rate], feed_dict=feed_dict)
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Learn rate: ", lr)

            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph dependencies.
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
        else:
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5 }
            _, l, predictions, merged_summary = session.run([optimizer, loss, train_prediction, merged], feed_dict=feed_dict)
            writer.add_summary(merged_summary, step)

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels))






