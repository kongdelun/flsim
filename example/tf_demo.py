import os

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# 一些参数
from tensorflow.python.keras.losses import categorical_crossentropy

from tensorflow.python.keras.utils.np_utils import to_categorical

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 128
epochs = 10
num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)  # 输入数据形状

# 获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 归一化
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 改变数据形状，格式为(n_samples, rows, cols, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# 控制台打印输出样本数量信息


# 样本标签转化为one-hot编码格式
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 创建CNN模型
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()  # 在控制台输出模型参数信息
model.compile(loss=categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 预测
n = 10  # 给出需要预测的图片数量，为了方便，只取前5张图片
predicted_number = model.predict(x_test[:n], n)
