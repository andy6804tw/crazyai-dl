---
title: 'DNN Classification'
description: ''
keywords: 
tags:
    - TensorFlow 101 從零開始
---

# 深度神經網路（DNN）分類基礎教學

範例程式：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andy6804tw/crazyai-dl/blob/main/code/tensorflow/DNN%20Classification.ipynb)

在這份教學中，我們將介紹如何使用 TensorFlow 來建構一個基礎的深度神經網路（DNN）進行分類任務。分類問題是機器學習中的一種重要任務，主要用於預測樣本屬於哪一個類別，例如圖片分類、垃圾郵件過濾等。在這裡，我們將一步步地帶領你學習如何從資料處理、模型構建到模型訓練，逐步掌握 DNN 分類的核心概念。

## 1. 載入套件

首先，我們匯入必要的套件：

```py
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## 2. 資料準備
在分類任務中，我們需要一組帶標籤的數據集。在這裡，我們將使用經典鳶尾花數據集（Iris Dataset），這是一個包含三個類別的數據集，用於預測鳶尾花的種類。

```py
# 載入鳶尾花數據集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 取得資料
iris = load_iris()
x = iris.data
y = iris.target

# 將數據分為訓練集和測試集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 標準化數據
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

這段程式使用 scikit-learn 來載入鳶尾花數據集，並將其分為訓練集和測試集，最後使用 `StandardScaler` 進行標準化處理，讓數據在訓練時能更快收斂。

## 3. 建構 DNN 模型
接下來，我們來建構一個簡單的深度神經網路模型。這個模型將包含兩層全連接層，用於分類三個鳶尾花的類別。

```py
# 建構 DNN 模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

在這裡，我們使用了 `tf.keras.Sequential` 來建立模型，包含兩層隱藏層，每層有 16 個神經元，並使用 ReLU 作激發函數。最後一層是輸出層，有三個神經元，使用 Softmax 激發函數來進行分類。損失函數選擇了 `sparse_categorical_crossentropy`，適合用於多類別分類問題。

## 4. 模型訓練
我們已經建構了模型，接下來我們將模型與數據進行訓練。這段程式將模型訓練 100 個 epoch，每次使用 10 個數據進行更新。

```py
# 訓練模型
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)
```

透過調整 epoch 和 batch_size 的值，我們可以控制模型訓練的速度和效果。

## 5. 模型評估與預測
訓練完模型後，我們可以使用它來進行預測，並評估模型的效果。這段程式碼首先評估了模型在測試數據上的表現，並使用訓練好的模型對新數據進行預測。

```py
# 評估模型效果
loss, accuracy = model.evaluate(x_test, y_test)
print(f'模型損失（Loss）：{loss}, 準確率（Accuracy）：{accuracy}')

# 進行預測
sample = np.array([[5.0, 3.6, 1.4, 0.2]])  # 一筆測試數據
predicted_class = model.predict(sample)
print(f'預測的類別：{np.argmax(predicted_class)}')
```

這段程式碼使用模型對測試數據進行評估，並計算損失和準確率，最後使用模型對新的數據樣本進行分類預測。

## 6. 模型保存與輸出
最後，我們可以將訓練好的模型保存起來，以便日後使用或部署。這段程式將模型保存為 SavedModel 格式，以便在生產環境中使用。

```py
# 保存模型
model.save('dnn_classification_model')
print('模型已成功保存至 dnn_classification_model')
```

!!! info

    TensorFlow 提供多種方法來儲存模型，以下是幾種常見的模型儲存方式：

    - **HDF5 格式（.h5 文件）**：適合在開發過程中使用，尤其是在多次迭代、快速測試模型或需要保存和重新載入模型進行訓練時。
    - **SavedModel 格式**：適合於模型部署與生產環境中使用。

    延伸閱讀： [Tensorflow Keras 模型儲存](https://andy6804tw.github.io/2021/03/29/tensorflow-save-model/)

在保存模型之後，我們可以重新載入它並進行推論。

```py
# 載入模型並進行推論
loaded_model = tf.keras.models.load_model('dnn_classification_model')
print('模型已成功載入')

# 使用載入的模型進行推論
predicted_class = loaded_model.predict(sample)
print(f'當輸入樣本為 {sample} 時，預測的類別為：{np.argmax(predicted_class)}')
```

!!! note

    除了使用 TensorFlow 內建的格式保存模型外，當模型訓練完成並準備正式部署於產品中時，也可以使用 ONNX（Open Neural Network Exchange）格式。ONNX 是一種開放格式，支持在不同深度學習框架之間進行模型轉換，使得模型可以通過 ONNX Runtime 在多種平台上執行，提高部署的靈活性。

## 結論
在本教學中，我們學會了如何使用 TensorFlow 構建一個簡單的深度神經網路來進行分類任務。我們從資料準備開始，一直到模型建構、訓練和評估，完整地了解了整個過程。接下來，你可以嘗試使用其他不同的資料集或改變模型結構，來進一步提高模型的準確度和效果。
