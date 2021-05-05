import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True,
                                                  normalize=True,
                                                  one_hot_label=True)

learning_rate = 0.02
EPOCH = 50
batch_size = 256
display_step = 1
example_to_show = 10
input_size = 784
hidden1_size = 256
hidden2_size = 128

train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.shuffle(60000).batch(batch_size)


def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)


class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(hidden1_size,
                                                   activation='sigmoid',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1())
        self.hidden_layer2 = tf.keras.layers.Dense(hidden2_size,
                                                   activation='sigmoid',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1())
        self.hidden_layer3 = tf.keras.layers.Dense(hidden1_size,
                                                   activation='sigmoid',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1())
        self.output_layer = tf.keras.layers.Dense(input_size,
                                                   activation='sigmoid',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1())


    def call(self, x):
        h1_output = self.hidden_layer1(x)
        h2_output = self.hidden_layer2(h1_output)
        h3_output = self.hidden_layer3(h2_output)
        reconstructed_x = self.output_layer(h3_output)

        return reconstructed_x


def mse_loss(y_pred, y_true):
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2))


optimizer = tf.optimizers.RMSprop(learning_rate)


def train_step(model, x):
    y_true = x
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mse_loss(y_pred, y_true)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


AutoEncoder_model = AutoEncoder()

for epoch in range(EPOCH):
    for batch_x in train_data:
        train_step(AutoEncoder_model, batch_x)
        current_loss = mse_loss(AutoEncoder_model(batch_x), batch_x)

    if epoch % display_step == 0:
        print('Epoch : %d\t Loss : %f' %( (epoch+1), current_loss))

reconstructed_result = AutoEncoder_model(x_test[:example_to_show])

fig, ax = plt.subplots(2, 10, figsize=(10, 2))
for i in range(example_to_show):
    ax[0][i].imshow(np.reshape(x_test[i], (28, 28)))
    ax[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))

fig.savefig('reconstructed_mnist_image.png')
fig.show()
plt.draw()




