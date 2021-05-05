import tensorflow as tf
from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True,
                                                  normalize=True,
                                                  one_hot_label=True)

learning_rate = 0.001
EPOCHS = 30
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)


def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)


class ANN(tf.keras.Model):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(hidden1_size,
                                                   activation='relu',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1())
        self.hidden_layer2 = tf.keras.layers.Dense(hidden2_size,
                                                   activation='relu',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1())
        self.output_layer = tf.keras.layers.Dense(output_size,
                                                  activation=None,
                                                  kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                  bias_initializer=random_normal_initializer_with_stddev_1())

    def call(self, x):
        h1_output = self.hidden_layer1(x)
        h2_output = self.hidden_layer2(h1_output)
        logits = self.output_layer(h2_output)

        return logits


def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


optimizer = tf.optimizers.Adam(learning_rate)


def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


ANN_model = ANN()

for epoch in range(EPOCHS):
    average_loss = 0.
    total_batch = int(x_train.shape[0] / batch_size)

    for batch_x, batch_y in train_data:
        _, current_loss = train_step(ANN_model, batch_x, batch_y), \
                          cross_entropy_loss(ANN_model(batch_x), batch_y)

        average_loss += current_loss / total_batch

    if epoch % display_step == 0:
        print('Epoch: %d\t Loss: %f' % ((epoch+1), average_loss))

print('Accuracy: %f' % compute_accuracy(ANN_model(x_test), y_test))




