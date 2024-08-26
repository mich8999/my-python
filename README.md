# my-python
深度學習的實作<br>

* neural network regression程式，是使用tensorflow套件對insurance和boston_housing兩個資料集，建置神經網路迴歸（neural network regression）模型，而資料中的自變項有類別型和數值型，依變項為數值型，再對資料集切為70%訓練集、30%測試集，最後將預測結果與損失函數（loss function）用圖片呈現。
* neural network classification程式，首先使用make_moons()創建資料集（2個數值型的自變項、1個類別型的依變項），對數值型的變項進行Min-Max標準化，接著將資料集切為70%訓練集、30%測試集，再以tensorflow套件建立預測模型（二元分類），正確率（accuracy）達0.873，最後以圖片呈現原始資料集中自變項與依變項的值；使用Fashion MNIST資料集（28x28像素），先將資料集切分為訓練集、測試集、以tensorflow套件建立預測模型（多元分類，10個），再以混淆矩陣評估預測模型，最後以圖片呈現預測結果與其他細節。


