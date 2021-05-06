import tensorflow as tf
from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True,
                                                  normalize=True,
                                                  one_hot_label=True)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(50)
train_data_iter = iter(train_data)


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=32,
                                                  kernel_size=5,
                                                  strides=1,
                                                  padding='same',
                                                  activation='relu')
        self.pool_layer1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                     strides=2)
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=5,
                                                  strides=1,
                                                  padding='same',
                                                  activation='relu')
        self.pool_layer2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                     strides=2)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer = tf.keras.layers.Dense(1024, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation=None)


    def call(self, x):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = self.conv_layer1(x_image)
        h_pool1 = self.pool_layer1(h_conv1)
        h_conv2 = self.conv_layer2(h_pool1)
        h_pool2 = self.pool_layer2(h_conv2)
        h_flat = self.flatten_layer(h_pool2)
        h_fc = self.fc_layer(h_flat)
        logits = self.output_layer(h_fc)
        # y_pred = tf.nn.softmax(logits)

        return logits


def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y))


optimizer = tf.optimizers.Adam(1e-4)


def train_step(model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


CNN_model = CNN()

for i in range(10000):
    batch_x, batch_y = next(train_data_iter)

    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_model(batch_x), batch_y)
        print('Epoch : %d\t Accuracy : %f' % (i, train_accuracy))

    train_step(CNN_model, batch_x, batch_y)

print('Total Accuracy : %f' % (compute_accuracy(CNN_model(x_test), y_test)))


