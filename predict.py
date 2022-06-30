import numpy as np
from pandas import DataFrame
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn import metrics
from tensorflow.keras.models import *

model = load_model('30/cnn.h5')
# model = load_model('50/cnn.h5')

test_dir = "./data/test"
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='binary',
    shuffle=False)  # 这里一定要设置成False,如果设置成True就会出现文中提到的现象


# 设置阈值，该概率大于0.5的类别为狗，即1
def pre_labels(y_pred):
    predicted_class = []
    for i in y_pred:
        if i > 0.5:
            label = 1
        else:
            label = 0
        predicted_class.append(label)
    return predicted_class


labels = test_generator.class_indices  # 查看类别的label，有什么标签
pred = model.predict(test_generator, verbose=1)  # 然后直接用predict_geneorator 可以进行预测
pred_label = pre_labels(pred)  # 调用函数
true_label = test_generator.classes  # 测试集的真实类别

#  画混淆矩阵
cm = confusion_matrix(true_label, pred_label, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.savefig('matrix.png')
plt.show()

report = metrics.classification_report(true_label, pred_label, target_names=['cat', 'dog'],
                                       output_dict=True)  # 获得分类报告
acc_report_df = DataFrame(report).T
acc_report_df.iloc[-3, :2] = np.nan
acc_report_df.iloc[-3, 3] = acc_report_df.iloc[-2, 3]
acc_report_df.to_csv("result.csv", index=True)

