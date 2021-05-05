import tensorflow as tf
import numpy as np
import pickle
from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True,
                                                  normalize=True,
                                                  one_hot_label=True)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data) # next호출 시 다음 배치 데이터셋 가져옴


class SoftmaxRegression(tf.keras.Model):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.softmax_layer = tf.keras.layers.Dense(10,
                                                   activation=None,
                                                   kernel_initializer='zeros',
                                                   bias_initializer='zeros')

    def call(self, x):
        logits = self.softmax_layer(x)

        return logits


def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))


optimizer = tf.optimizers.SGD(0.5)


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


SoftmaxRegression_model = SoftmaxRegression()

for i in range(1000):
    batch_x, batch_y = next(train_data_iter)
    train_step(SoftmaxRegression_model, batch_x, batch_y)

print('Accuracy : %f'
      % compute_accuracy(tf.nn.softmax(SoftmaxRegression_model(x_test)), y_test))



