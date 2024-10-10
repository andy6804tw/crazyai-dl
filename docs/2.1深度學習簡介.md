---
layout: post
title: '[AI學習筆記] 李宏毅課程 - 深度學習簡介'
categories: 'AI'
description:
keywords: Deep Learning 
---

## 前言
深度學習是一種機器學習的方法。它藉由模仿人類大腦神經元的結構，定義解決問題的函式。所謂深度學習是一種具有深度多層的神經網路。機器可以自行學習並且理解機器學習時用以表示資料的「特徵」，因此又稱為「特徵表達學習」，其應用包括：影像分類、機器翻譯...等。

![](https://ithelp.ithome.com.tw/upload/images/20210921/20107247LthVZkdrnv.png)

# Deep Learning
我們稍微回顧一下深度學習的歷史。他在歷史上經過好幾次的起起伏伏。首先在 1958 年 Perceptron 觀念被提出，它也是一個線性的模型。起初由 Frank Rosenblatt 在海軍的專案裡被提出來的，在當時 Perceptron 開啟了人工智慧機器自己學習的開端。後來有人指出線性的模型是有極限的，無法解決的多現實生活中許多問題。後來就有人提出多個 Perceptron 的 Multi-layer perceptron，並與現今的大家所用的 DNN 架構差不多。其中在 1986 年 Hinton 提出了 Backpropagation 的學習機制，但是當時僅限於三層的神經網路架構。然而在 1989 年又有人提出其實一層的 hidden layer 其實夠夠用了，因此那段時期大家都喜歡使用 SVM 而神經網路再度的被遺棄。直到 2006 年 Hinton 使用 Restricted Boltzmann Machine (RBM) 受限玻爾茲曼機 做神經網路的  initialization 稱作深度學習。RBM 使用比較深的理論並採用 graphical 模型並非現今大家所用的 neural nework。最後大家才發現原來其實這個方法其實也沒什麼用，如果你去讀深度學習的文獻現在已經沒有人用 RBM 做 initialization 了。不過它的出現讓大家再次對 Multi-layer perceptron 產生了興趣，因此 Google 的人還對 RBM 評論稱說這個方法就是石頭湯裡面的石頭。另外在 2009 年發現能夠用 GPU 加速運算，原本 CPU 需要花好幾週的運算透過 GPU 只需要幾小時就能完成了。另外在 2011 年神經網路開始運用在語音辨識，並發現結果非常好，因此大家瘋狂使用深度學習的技術。到了 2012 年的時候深度學習的技術贏了一個很重要的影像辨識的競賽，從此刻起影像的領域也開始使用了深度學習的技術。

![](https://i.imgur.com/rTY3Cxx.png)

下圖是google專案中應用到深度學習的趨勢圖，從2012年幾乎0到2016年逐步成長。

![](https://i.imgur.com/RqjLG7e.png)

其實深度學習的技術並沒有那麼複雜，它其實非常簡單。我們都知道訓練機器學習間單來說就是三個步驟，深度學習也如此。在深度學習中要找到的 function 其實就是一個神經網路。

![](https://i.imgur.com/w6V6ZHo.png)

我們把 Logistic Regression 前後相連再一起。將一個 Logistic Regression  稱為 Neuron。我們用不同的方法連接這些神經網路就會得到不同的架構。在這個神經網路裡面我們有一堆的 Logistic Regression ，每一個都有各自的權重與 bias。這些權重與 bias 集合起來就是神經網路的參數。

![](https://i.imgur.com/JTSQ9mK.png)

這些神經元該如何連接起來呢？最常見的做法是 Fully Connect Feedforward Network 全連接的方式將每個神經元彼此連接。所以一個神經網路你就可以把它當作是一個 function，如果一個神經網路裡面的參數 weight 和 bias 都得到的話，它就是一個 function。 

![](https://i.imgur.com/hYfESZq.png)

一個網路如果我們已經把參數設定上去的話，它就是一個 function。如果我們還不知道參數，僅是設計出網路的架構決定了這些神經元個數與隱藏層數量，其實就是定義了一個 function set。我們可以給這個網路不同的參數，雞漚會變成不同的 function。把這些可能的 function 通通集合起來，我們就得到了一個 function set。

![](https://i.imgur.com/HYZp2aT.png)

我們可以將神經網路表示成這個架構，擁有很多層的網路每一層擁有多個神經元。而途中每一顆圓形都表示一個神經元，每一層網路的神經元個數可以自行定義數量。每一層間的神經元是兩兩互相連接的，Layer 1 的輸出會直接給 Layer 2 每一個神經元。那 Layer 2 每個神經元的輸入就是 Layer 1 的輸出。因為 Layer 和 Layer 間所有的神經元兩兩相接，因此我們稱它 Fully Connected Network(全連接網路)。

整個網路需要一個輸入層，這個輸入就是一個向量。對於 Layer 1 的每一個神經元來說，每一個神經元它的輸入就是輸入層的維度。那最後第 L 層的那些神經元後面沒有接其他東西了，所以它的輸出就是整個網路的輸出就是所謂的輸出層。輸入的地方我們稱作 Input Layer，輸出的地方稱作 Output Layer，其餘的中間部分稱作 Hidden Layer 隱藏層。


![](https://i.imgur.com/wf5HXPy.png)

在 Deep Learning 中其中的 Deep 就是有很多的隱藏層。在 2012 年時候參加 imageNet 比賽得到冠軍的 AlexNet 他有 8 層錯誤率是 16.4%，在當年競賽第一二名差距懸殊，其第二名的錯誤率是 30%。到了 2014 年時候 VGG 有 19 層的網路，錯誤率降到 7%。此外 GoogleNet 擁有 22 層並擁有 6.7% 的錯誤率。但是這些都還不算什麼，直到 2015 年有了 152 層的 Residual Net 他的錯誤率降到了 3.57% 是個重大突破。然而這裡的 Residual 網路並非一般的全連階層網路，他透過特別的架構才能使得有這麼深層的網路同時不會過度擬合。

![](https://i.imgur.com/irDkY5W.png)

我們可以將神經網路中間的隱藏層視為一個特徵萃取器，就好比機器學習中的特徵工程。

![](https://i.imgur.com/g6fYnbO.png)

在神經網路中如何決定一組參數的好壞呢？我們必須設計一個損失函數從真實答案與預測結果比較出來之間的差異。並透過計算誤差來更新模型預測的方向，使其預測結果更貼近真實的目標答案。神經網路中的參數更新的方式就是採用到傳遞演算法，並透過梯度下降找到最佳的一組參數。實際上在深度學習裡面用的 Gradient Descent 跟 Linear Regression 那邊沒有什麼差別，兩者是一模一樣的。也就是說我們有一堆參數 𝜃，首先為每個參數隨機找一個初始值。接下來去計算他的梯度，計算每一個參數對 total loss 的偏微分。 最後把這些偏微分全部集合起來稱作梯度，有了這些偏微分以後我們就可以更新這些參數了。把所有的參數減去一個學習速率乘上偏微分的值稱作梯度，我們就能得到一組新的參數。上述的流程反覆進行下去，有了新的參數再去計算一下它的梯度，再根據新的梯度更新一次參數又會得到另一組新的參數。不斷的迭代下去，最終會收斂得到一組好的參數。以上就是整個神經網路的訓練機制。

![](https://i.imgur.com/EyjFumX.png)

現今有許多的深度學套件能夠幫助我們計算梯度與學習，即使你不會算微分也能透過這些工具實作神經網路。Backpropagation 是一個比較有效率的計算微分的方式。

![](https://i.imgur.com/hvP9pO3.png)

你可能會問，為何要用 Deep Learning?你可能會認為網路越深效果越好。下圖是一個早期 2011 年的實驗，隨著神經網路層數加深Word Error  Rate會越來越低。其原因是模型擁有越多的參數，它覆蓋的 function set 越大使得 bias 越小。如果我們有夠多的訓練資料去控制他的 variance，一個複雜的模型他的表現會比較好是正常的。

![](https://i.imgur.com/UJiZTyY.png)

甚至有一個理論說。假設任何連續的 function 它的輸入是一個 N 為的向量，輸出是一個 M 維的向量。它都可以用一個隱藏層的神經網路來表示。只要你的這一個隱藏層的神經元夠多，它可以表示成任何的 function。

![](https://i.imgur.com/nGBUQQZ.png)


- [簡報](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/DL%20(v2).pdf)
- [影片](https://www.youtube.com/watch?v=Dr-WRlEFefw&feature=youtu.be)
