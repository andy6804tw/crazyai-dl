---
title: 'TensorFlow 基礎介紹'
description: ''
keywords: 
---

# TensorFlow 基礎介紹

範例程式：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andy6804tw/crazyai-dl/blob/main/code/tensorflow/TensorFlow%20基礎介紹.ipynb)

TensorFlow 是一個由 Google 開發的機器學習平台，主要用於深度學習應用的開發和部署。它提供了豐富的工具集，從基礎層級的數學運算（如矩陣運算）到高層次的神經網絡構建，無論你是剛入門的初學者還是有經驗的研究者，TensorFlow 都是一個強大且靈活的工具。

TensorFlow 能夠讓開發者方便地構建、訓練和部署深度學習模型。它被廣泛應用於影像分類、語音辨識、自然語言處理等領域，並支持多種計算平台，包括 CPU、GPU、以及 TPU，讓運算的效能能夠靈活擴展。

首先，我們需要安裝並匯入相關套件。如果你還沒有安裝 TensorFlow，可以使用以下指令來安裝：

```sh
pip install tensorflow
```

## 1. 張量（Tensor）
在 TensorFlow 中，張量（Tensor）是進行所有運算的基本單位。張量的概念與 NumPy 陣列類似，但張量具備更強的擴展性，特別適合 GPU 加速的運算需求。

### 1.1 張量的基本操作

我們可以使用 `tf.constant()` 來創建一個靜態張量，或者使用 `tf.Variable()` 來創建一個可變張量。

```py
import tensorflow as tf
import numpy as np

# 建立一個靜態張量
a = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
print(a)

# 建立一個 NumPy 陣列並轉換為張量
b = tf.constant(np.array([1, 2, 3, 4, 5]), dtype=tf.float32)
print(b)

# 建立一個可變張量
c = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
print(c)
```

!!! note

        TensorFlow 張量和 NumPy 陣列不同之處在於，張量可以分為兩種：一種是**無法更改值的 Tensor**，另一種是**可以更改值的 Variable**。在神經網路中，所有可訓練的變數都以 Variable 形式存在，這樣才能在訓練過程中進行數值更新。

### 1.2 張量與 NumPy 的互動
張量可以輕鬆地與 NumPy 陣列相互轉換，這對於需要在 TensorFlow 與其他數值計算工具之間交互時非常有用：

```py
# 將 TensorFlow 張量轉換為 NumPy 陣列
d = a.numpy()
print(d)

# 將 NumPy 陣列轉換為張量
e = tf.convert_to_tensor(np.array([5, 4, 3, 2, 1]), dtype=tf.float32)
print(e)
```


### 1.3 張量的性質
張量具有以下幾個重要的性質：

1. **形狀（Shape）**：張量的形狀代表其在每個維度上的大小。形狀可以通過 `tensor.shape` 來獲取，這對於理解張量結構非常重要。
2. **資料型別（Data Type）**：張量中的每個元素都具有相同的資料型別，例如 float32、int32 等，可以通過 `tensor.dtype` 來獲取。
3. **設備（Device）**：張量可以被分配到不同的設備上進行計算，例如 CPU 或 GPU，可以通過 `tensor.device` 查看張量所在的設備。


```py
tensor = tf.constant([[1, 2], [3, 4], [5, 6]])
print(tensor.shape)  # 結果為 (3, 2)，表示張量有 3 個列和 2 個行
```

```py
tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
print(tensor.dtype)  # 結果為 float32
```


```py
with tf.device('/CPU:0'):
    tensor = tf.constant([1, 2, 3])
print(tensor.device)  # 顯示張量所在的設備，例如 '/job:localhost/replica:0/task:0/device:CPU:0'
```

### 1.4 張量的數學運算
張量支援多種數學運算，例如加減乘除、矩陣乘法、指數運算等，這些操作可以非常簡單地透過 TensorFlow 提供的函數來實現。

以下是一些常見的張量運算範例：

```py
# 建立兩個張量
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# 張量加法
c = tf.add(a, b)
print(c)  # 結果為 [[ 6  8]
          #        [10 12]]

# 張量減法
d = tf.subtract(a, b)
print(d)  # 結果為 [[-4 -4]
          #        [-4 -4]]

# 張量乘法（逐元素相乘）
e = tf.multiply(a, b)
print(e)  # 結果為 [[ 5 12]
          #        [21 32]]

# 矩陣乘法
f = tf.matmul(a, b)
print(f)  # 結果為 [[19 22]
          #        [43 50]]
```

## 2. 自動計算微分值
在深度學習演算法中，模型的權重更新是非常重要的環節，這需要對變數進行偏微分計算。
TensorFlow 提供了一個強大的工具來進行自動微分，即 `tf.GradientTape()`。這使得神經網路的訓練過程變得非常簡單，因為你可以輕鬆地計算導數並更新模型參數。

### 2.1 使用 `tf.GradientTape()` 進行微分
以下是如何使用 `tf.GradientTape()` 來進行自動微分的例子：

!!! note

        若此函數 $f(x) = x^2$ 對 $x$ 做偏微分，則能得到 $f^\prime(x) = 2*x$

        將 $x = 3$ 代入函數，得到 $f(x)=9$，$f^\prime(x) = 6$


```py
# 自動微分範例
x = tf.Variable(3.0)

def f(x):
    return x**2

with tf.GradientTape() as tape:
    y = f(x)

dy_dx = tape.gradient(y, x) 
print(dy_dx)  # 結果應該是 6.0，因為 y = x^2 對 x 的導數為 2*x
```

在上述程式碼中，我們展示了如何使用 TensorFlow 的 `tf.GradientTape()` 進行自動微分。首先，我們定義了一個變數 x，並將其初始值設定為 3.0。接著，我們定義了一個函數 f(x)，它返回 x 的平方。在 `with tf.GradientTape() as tape:` 這段程式中，我們計算了 f(x)，即 `y = x**2`。使用 `tape.gradient(y, x)`，我們計算出 y 對 x 的導數，結果為 6.0，因為 `y = x^2` 對應的導數為 `2*x`，當 x=3 時，導數的值為 6.0。

!!! note

        這種自動微分的方式，對於計算複雜神經網路模型的梯度尤為重要，讓我們可以在訓練過程中更新權重，最小化損失。


## 3. 模型建置以及訓練
在 TensorFlow 中，我們可以通過高階 API（如 tf.keras）來快速構建和訓練模型。

### 3.1 使用 Keras 來建構模型
以下是一個簡單的線性迴歸模型的構建和訓練過程。這段程式碼展示了如何使用 tf.keras 構建並訓練一個簡單的線性迴歸模型，包括如何編譯模型、設定優化器和損失函數、提供訓練數據，以及最終進行預測。

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 建立一個簡單的順序模型
model = Sequential([
    Dense(1, input_shape=(1,), activation='linear')
])

# 編譯模型，設定損失函數和優化器
model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')

# 建立一些訓練數據
X_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# 訓練模型
model.fit(X_train, y_train, epochs=100)

# 使用模型進行預測
print(model.predict([6]))  # 應該接近於 12
```


## 4. TensorFlow Keras 網路搭建的三種方法
在 TensorFlow 中使用 Keras 來搭建神經網路有多種方法，根據需求的靈活性、網路的結構以及建模的複雜度，以下介紹三種主要的方式：

### 4.1 使用 Sequential API
Sequential API 是最簡單和直觀的方法之一，適合用來搭建「線性堆疊」的模型，也就是一層一層按順序疊加的結構。

- **優點**：簡單易用，適合快速原型設計和初學者。
- **缺點**：只適合用來構建單一路徑的網路，無法處理複雜的結構（例如分叉或多輸入/多輸出網路）。

**範例**：

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 4.2 使用 Functional API
Functional API 提供了更多的靈活性，適合構建更複雜的網路，例如具有多輸入、多輸出或跳接連接的網路。

- **優點**：靈活度更高，可以自由定義多輸入、多輸出、跳躍連接等結構。
- **缺點**：相較於 Sequential API 稍微複雜一點，適合需要更高自由度的模型設計。

**範例**：

```py
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(input_dim,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
```

!!! note

        這種方式允許構建更複雜的結構，例如 ResNet 這類有「殘差」連接的網路。

### 4.3 繼承 Model 類別（Subclassing API）
通過繼承 tf.keras.Model 類別來自定義網路，這是最靈活的方法，能夠完全控制模型的結構和前向傳播的過程。

- **優點**：適合非常複雜且需要對前向傳播進行細緻控制的模型設計。
- **缺點**：比起 Sequential 和 Functional API 更加難以調試和管理，特別是當模型變得非常大時。

**範例**：

```py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.out = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

model = MyModel()
```

!!! note

        在這裡，call 方法定義了前向傳播的邏輯，因此能夠完全掌控如何處理輸入和輸出，適合需要對模型進行特別處理的場合，例如自定義的訓練。

#### 綜合比較
選擇哪一種方法取決於使用情境：如果只是構建一個基礎的神經網路，Sequential 最方便；如果需要更靈活的網路結構，Functional API 非常合適；而當需要完全掌控模型時，特別是演算法研究與開發，則應該考慮繼承 Model 類別的方法。

1. **Sequential API**：適合初學者和簡單的線性堆疊模型。
2. **Functional API**：適合需要更靈活的網路拓撲結構，如分叉結構、多輸入或多輸出等。
3. **Model 繼承 (Subclassing API)**：適合最複雜的場景，完全自定義模型行為。