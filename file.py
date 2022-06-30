import os
import shutil

original_dataset_dir = "C:/Users/FANGJIAMIAN/Desktop/猫和狗/train"
# 保存较小的数据集
train_cats_dir = "./data/train/cats"
validation_cats_dir = "./data/validation/cats"
test_cats_dir = "./data/test/cats"
train_dogs_dir = "./data/train/dogs"
validation_dogs_dir = "./data/validation/dogs"
test_dogs_dir = "./data/test/dogs"

filenames = ['cat.{}.jpg'.format(i) for i in range(7500)]  #
for filename in filenames:
    # 前1000张猫头像复制到train_cats_dir目录下
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(train_cats_dir, filename)
    shutil.copyfile(src, dst)
filenames = ['cat.{}.jpg'.format(i) for i in range(7500, 10000)]  #
for filename in filenames:
    # 接下来500张猫头像复制到validation_cats_dir目录下
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(validation_cats_dir, filename)
    shutil.copyfile(src, dst)
filenames = ['cat.{}.jpg'.format(i) for i in range(10000, 12500)]  #
for filename in filenames:
    # 接下来500张猫头像复制到validation_cats_dir目录下
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(test_cats_dir, filename)
    shutil.copyfile(src, dst)

# 狗类别同上操作
filenames = ['dog.{}.jpg'.format(i) for i in range(7500)]  #
for filename in filenames:
    # 前1000张猫头像复制到train_dogs_dir目录下
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(train_dogs_dir, filename)
    shutil.copyfile(src, dst)
filenames = ['dog.{}.jpg'.format(i) for i in range(7500, 10000)]  #
for filename in filenames:
    # 接下来500张猫头像复制到validation_dogs_dir目录下
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(validation_dogs_dir, filename)
    shutil.copyfile(src, dst)
filenames = ['dog.{}.jpg'.format(i) for i in range(10000, 12500)]  #
for filename in filenames:
    # 接下来500张猫头像复制到validation_cats_dir目录下
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(test_dogs_dir, filename)
    shutil.copyfile(src, dst)
