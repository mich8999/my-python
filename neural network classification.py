import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images , train_labels) , (test_images , test_labels) = fashion_mnist.load_data() #資料集已被設定好比例
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape , train_labels.shape , test_images.shape , test_labels.shape)


def softmax(x):

  x = tf.cast(x , dtype = tf.float32) #將x轉換為float32型態

  e_x = tf.math.exp(x - tf.math.reduce_max(x)) #減去最大值，避免溢位
  return e_x / tf.math.reduce_sum(e_x , axis = 0) #進行標準化，將每column的每個元素除以該column的總和，將其轉換為機率，這樣每個元素的值都在0和1之間，並且所有元素的總和為1


#進行測試
tensor = tf.constant([[1, 2, 3, 6],
            [2, 4, 5, 6],
            [3, 8, 7, 6]] )
print(tensor)
print(softmax(tensor))

import keras
from keras import layers 

simple_model = tf.keras.Sequential([
  layers.Flatten(input_shape = (28 , 28)),
  layers.Dense(64 , activation= 'relu'),
  layers.Dense(32 , activation = 'relu'),
  layers.Dense(10 , activation='softmax') #有10個類別
])

simple_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), #多元分類
                optimizer = tf.keras.optimizers.Adam() ,
                metrics = ['accuracy'])


history = simple_model.fit(train_images , train_labels , epochs = 50)


import itertools
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
  #標準化混淆矩陣，將每個元素除以其所在row的總和，得到每個元素在其所在row的比例。
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] #增加一個維度，將row總和轉換為一個column向量
  n_classes = cm.shape[0] #表示混淆矩陣的類別數

  fig, ax = plt.subplots(figsize=figsize) #fig為大圖、ax為子圖
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes),
         xticklabels=labels,
         yticklabels=labels)

  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  
  plt.xticks(rotation=20)

  threshold = (cm.max() + cm.min()) / 2. #計算閾值

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)
    

pred_probs = simple_model.predict(test_images)
preds = pred_probs.argmax(axis = 1) #返回的是pred_probs中每一row中最大值的索引，即每個樣本之預測機率最大的類別索引
print(preds[:10]) #顯示前10個資料點是甚麼類別

make_confusion_matrix(y_true = test_labels,
           y_pred = preds,
           classes = class_names,
           figsize = (20 , 15),
           text_size = 5)

preds = simple_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i] #i表示圖片的索引
  plt.grid(False) #不顯示圖片的網格線
  plt.xticks([]) #隱藏x軸和y軸的刻度
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary) #圖片以灰度（binary）方式顯示

  predicted_label = np.argmax(predictions_array) #取得預測結果中機率最高的類別
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                      100*np.max(predictions_array), #將預測的機率轉為百分比
                      class_names[true_label]),
                      color=color)
  #預測類別名稱 預測百分比% (真實類別名稱)，舉例Bag 91% (Sneaker)，表示預測圖片為Bag機率為91%

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10)) #設置x軸刻度為0到9（依照類別設置）
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red') #將預測標籤的長條圖顯示為紅色，表示模型的預測
  thisplot[true_label].set_color('blue') #將真實標籤的長條圖顯示為藍色，表示正確的值
  #灰色：表示其他類別的預測機率，這些類別既不是模型的最終預測，也不是實際的類別


def plot_prediction_images():
  #表示5x3個圖片
  num_rows = 5
  num_cols = 3
  num_images = num_rows * num_cols
  plt.figure(figsize = (2*2*num_cols , 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows , 2*num_cols , 2*i+1)
    plot_image(i , preds[i], test_labels , test_images)
    plt.subplot(num_rows , 2*num_cols , 2*i+2)
    plot_value_array(i , preds[i] ,  test_labels)
  plt.suptitle("Prediction outcome")
  plt.tight_layout() #自動調整子圖的佈局，避免重疊、擁擠
  plt.show()


plot_prediction_images()
