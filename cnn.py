from matplotlib import pyplot as plt
from pandas import DataFrame
from tensorflow import optimizers
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import *

# 训练样本目录和验证样本目录
train_dir = './data/train'  # 训练集20000条，猫狗各一半
validation_dir = './data/validation'  # 验证集5000条，猫狗各一半


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    model = build_model()
    # 配置训练器,使用二元交叉熵作为损失函数
    optimizer = optimizers.RMSprop(lr=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # 数据增强，增加数据量，让有限的数据产生等价于更多数据的价值
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,  # 随机旋转角度范围
                                       width_shift_range=0.2,  # 相对于总宽水平移动比例
                                       height_shift_range=0.2,  # 相对于总高度0水平移动比例
                                       shear_range=0.2,  # 随机错切角度
                                       zoom_range=0.2,  # 随机缩放范围
                                       horizontal_flip=True,  # 随机将一半图片翻转
                                       fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'  # 损失函数是binary_crossentropy 所以使用二进制标签
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )
    history = model.fit(
        # 读取数据生成器中数据进行拟合
        train_generator,
        steps_per_epoch=120,  # 生成器返回次数
        epochs=50,
        validation_data=validation_generator,
        validation_steps=40
    )
    hist_df = DataFrame(history.history)
    hist_df.to_csv('history.csv', index=False)

    history_dict = history.history  # 字典形式

    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    epochs = range(1, len(acc) + 1)

    # acc
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and Validation acc")
    plt.legend()  # 显示图例名称
    plt.savefig('acc.png')
    plt.figure()

    # loss
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()  # 显示图例名称
    plt.savefig('loss.png')
    plt.show()

    model.save('cnn.h5')
