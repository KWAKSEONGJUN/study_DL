import tensorflow as tf

# 선형회귀 모델 (wx + b)
w = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))


# 가설 정의
# 예측 y값 출력
def linear_model(x):
    return w*x + b


# 손실 함수 정의
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))


# optimizer 정의
optimizer = tf.optimizers.SGD(0.01)


# 최적화를 위한 function 정의
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = linear_model(x)
        loss = mse_loss(y_pred, y)
    gradients = tape.gradient(loss, [w, b])
    optimizer.apply_gradients(zip(gradients, [w, b]))


x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

for i in range(1000):
    train_step(x_train, y_train)

x_test = [3.5, 5, 5.5, 6]
print(linear_model(x_test).numpy())
print(w, b)






