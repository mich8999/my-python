
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv(
    'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(data.shape)
print(data.head())


# 變項中有幾種不同的值
unique_values = data['region'].unique()
print(unique_values)


# 重新定義變項
data2 = pd.DataFrame(data)
data2['sex'] = data2['sex'].map({'male': 1, 'female': 2})
data2['smoker'] = data2['smoker'].map({'no': 0, 'yes': 1})
data2['region'] = data2['region'].map(
    {'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4})
print(data2)


# 分割x和y，y為醫療保險的金額
x = data2.drop('charges', axis=1)
y = data2['charges']
print(x.shape, y.shape)


# 分為訓練集70%和測試集30%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1234)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


tf.random.set_seed(1234)
# 建置模型
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(300),
    tf.keras.layers.Dense(200),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])
# 編譯模型
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])


sys.stdout.reconfigure(encoding='utf-8')
# 訓練模型
history = model_1.fit(x_train, y_train, epochs=100, verbose=2)
# 評估模型
model_1.evaluate(x_test, y_test, verbose=2)
# 模型預測
y_preds_1 = model_1.predict(x_test)
print(y_preds_1)


# 可視化模型（定義）
def plot_predictions(x_train, y_train, x_test, y_test, predictions):
    plt.figure(figsize=(7, 7))

    plt.scatter(x_train, y_train, c="b", label="Training data") #用藍色表示
    plt.scatter(x_test, y_test, c="g", label="Testing data") #用綠色表示
    plt.scatter(x_test, predictions, c="r", label="Predictions") #用紅色表示

    plt.xlabel('Feature - age')
    plt.ylabel('Target - charges')
    plt.title('Model outcome')
    plt.legend(); #表示圖例說明
    plt.show()


# 套用上面的定義，選取一個自變項（age）的索引進行可視化呈現，0表示column的索引位置，也就是第一欄
plot_predictions(x_train.iloc[:, 0], y_train,
                 x_test.iloc[:, 0], y_test, y_preds_1)


# 繪製模型訓練過程中的損失曲線
pd.DataFrame(history.history).plot()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()