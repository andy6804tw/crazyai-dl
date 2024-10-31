
# 影像分割必讀論文 Fully Convolutional Networks
> 全卷積神經網路
## 前言
根據統計近年來影象分割技術基於深度學習都是基於一個共同的先驅，就是本篇要提的 FCN（Fully Convolutional Network，全卷積神經網路）。電腦視覺中最重要的三個任務分別有影像識別、物件偵測以及本文要討論的影像分割。Semantic segmentation 的目標是，輸入一張彩色的影像並且以像素為單位去識別影像中的每個 pixel 是屬於事先定義好的類別中的哪一類或是背景。為何這個議題非常重要呢？其原因是人類的視覺系統中可以辦到以像素為單位去辨識影像中的物體。現實生活中影像分割的例子有：自駕車、醫學影像…等技術都是建立在 Semantic segmentation 之上。

![](https://i.imgur.com/0DFFTKs.png)

- Semantic Segmentation: 是指將圖像中的所有像素點進行分類。
- Instance Segmentation: 是物件偵測和語義分割的結合。

本篇要討論的論文全名為 Fully Convolutional Networks for Semantic Segmentation，發表在2015的CVPR上。

- 論文：https://arxiv.org/abs/1411.4038

## FCN 主要解決什麼問題？
我們都知道在深度學習中卷積神經網路的出現，使得我們在影像識別的應用上達到非常強大的效果。若以 Semantic segmentation 來看的確也是一個分類的問題，但是它是以一張圖中每個像素為單位進行分類。若我們將每一個 pixel 都依序丟入一個訓練好的 CNN 分類器雖然也能達到想要的效果，但是此做法非常沒效率。

![](https://i.imgur.com/blbHHcW.png)

因此我們要來探討目前在語意分割技術上最常見的做法就是 Fully convolutional，完全使用卷積來完成語意分割的任務。因為語意分割是一個 pixels in, pixels out 的應用，假設我輸入是一個 `H*W*3` 大小的圖，輸出的結果也應該是以像素為單位 `H*W` 的圖。我們可以透過 padding 來確保每一層卷積的輸出大小一致，所以最終的輸出也能得到和原始大小一樣的圖。這裡值得一提的是，假設我們有 C 類的物件要辨識，所以我們在最後一層卷積輸出的時後我們希望他有 C 張特徵圖，而每一張的特徵圖(`H*W`)就是對應每個 pixel 在某類別上的機率是多少。但是如果圖片都是高解析度且是一個非常深層的網路的時候還是一個非常耗時的模型，因為每一層網路都是一樣的 `H*W` 大小的圖片計算量非常大。

![](https://i.imgur.com/DRR1NQ5.png)

所以接下來演變成如圖中的作法，雖然保留了先前作法但我們可以發現它很像是一個 AutoEncoder 編碼器的架構。前面的網路是做 downsampling 解析度越來越低降為的動作，後面的網路是 upsampling 從低解析度升回去原來的解析度。我們可以從這個架構看到中間的網路層圖片的解析度是適度的壓縮。實驗證明這樣子的架構不僅效率變好，甚至效果也比較好。首先在 Downsampling 方法中我們可以透過 Pooling 或 strided 將一張特徵圖(feature map) 的解析度越變越低。

![](https://i.imgur.com/lYXrnw8.png)

## Fully Convolutional Networks (FCN)
因為模型網路中每一層都是卷積層，故稱為全卷積神經網路(FCN)。FCN 在最後幾層網路用卷積層代替傳統的全連接層，使得輸出也可以變成二維形式，這一做法使得輸出中保留了空間訊息，對於語義分割這種任務來說還是非常有效的，同時還透過反卷積解決了輸入圖片尺寸的問題。全卷積神經網路主要使用了下面兩個技巧：

1. Downsampling (Pooling 與 strided convolution)
2. Upsampling (Unpooling 與 Transposed convolution)

在早期 2015 年前的研究是將經典 CNN 模型放在前面，將一張影像進行特徵萃取最後一層透過 1*1 卷積輸出成 classes+1(圖中的21=20類+背景) 的 channel 維度。如何將低解析度的特徵圖還原到高解析度是這個問題的最大挑戰，同時這也是 FCN 的最大貢獻。簡單來說作者透過 Upsampling 來實現從低解析度到高解析度的輸出，真正的背後技術就是 Transposed convolution。但是此方法發現進行 deconvolution 效果是有限的，因此在 2015 年 FCN 提出一個機制來達到 Upsampling 的整個優化過程。


![](https://i.imgur.com/Q9cFPxZ.png)

以下是作者提出的 FCN 架構，首先輸入一張影像並透過 pooling 或是 stride convolution 將解析度逐漸變低。假設在第五個 block 輸出的解析度只有原先影像的 1/32。先前有提到 deconvolution 可以提高解析度。因此第一版 FCN-32s 做法是直接將 block5 的輸出做 deconvolution 得到原來大小，但效果有限預測出來的邊界模糊。那個該如何解決這個問題就是 FCN 這篇論文的貢獻。如果我們將 block5 的 1/32 影像先 upsampled 變成兩倍，此時的解析度跟 block4 一樣了。當一樣的解析度隨然可能會有不同的意義，但是它指的位置都是類似的。因此這裡做一個假設將兩個解析度一樣的圖作加總，之後再做 16 倍的 upsampled 透過 deconvolution 得到的結果為 FCN-16s。依此類推我們可以用同樣方法得到更精確的 FCN-8s 也是最終論文 FCN 的版本，用了兩次 skip connection 藉由 pool3 和 pool4 的特徵圖再經過八倍的 deconvolution 得到一個較好的高解析度輸出。

![](https://i.imgur.com/4ETid3Z.png)

個人認為 FCN 這篇論文的主要貢獻是使用反卷積並從前幾層的池化和卷積資訊結合，結合歷史資訊填補我們缺失的資料。

![](https://i.imgur.com/buZPxuZ.png)

## Reference
- [語意分割演進](https://www.gushiciku.cn/dc_tw/109328837)