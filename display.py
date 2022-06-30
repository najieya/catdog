import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def get_img(img_path):
    # 根据路径读取图片
    img = tf.io.read_file(img_path)
    # 解码图片，这里应该是解码成了jpg格式
    img = tf.image.decode_jpeg(img, channels=0)
    # 大小缩放
    img = tf.image.resize(img, [150, 150])
    # 转换成张量
    img = tf.cast(img, dtype=tf.float32) / 255.
    img = tf.expand_dims(img, 0)
    return img


def get_type(result1):
    if result1 > 0.5:
        return 'dog'
    else:
        return 'cat'


predict_model = tf.keras.models.load_model('50/cnn.h5')

print("--------------------")
imgs = [get_img('./data/display/' + str(i + 1) + '.jpg') for i in range(11)]
imgs = tf.concat(imgs, axis=0)
results = predict_model.predict(imgs)

plt.figure()
for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(imgs[i - 1])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(get_type(results[i]))
plt.show()

