

CNN 的強大之處在於它的多層結構能自動學習特徵，並且可以學習到多個層次的特徵：較淺的卷積層感知域較小，學習到一些區域性區域的特徵。較深的卷積層具有較大的感知域，能夠學習到更加抽象一些的特徵。這些抽象特徵對物體的大小、位置和方向等敏感性更低，從而有助於識別效能的提高。

## 名詞
- 卷積核(Kernel)又稱為Filters
    - 我們可以藉由選擇Kernel的張數控制 Feature maps

## 影像的上取樣
影像的上取樣正好和卷積的方式相反，我們可以通過正常的卷積讓影像越來越小，而上取樣則可以同樣通過卷積將影像變得越來越大，最後縮放成和原影像同樣大小的圖片。上取樣有3種常見的方法：雙線性插值(bilinear)，反摺積(Transposed Convolution)，反池化(Unpooling)。

我們先來回顧一下正向卷積，也稱為下采樣，正向的卷積如下所示，首先我們擁有一個這樣的 3*3 的卷積核：

![](https://i.imgur.com/sppZpw2.png)

然後對一個`5*5`的特徵圖利用滑動視窗法進行卷積操作，padding=0,stride=1,kernel size=3,所以最後得到一個`3*3`的特徵圖：

![](https://i.imgur.com/CXNiU0A.png)

那麼上取樣呢？則是這樣的，我們假定輸入只是一個`2*2`的特徵圖，輸出則是一個`4*4`的特徵圖，我們首先將原始`2*2`的map進行周圍填充pading=2的操作，筆者查閱了很多資料才知道，這裡周圍都填充了數字0，周圍的padding並不是通過神經網路訓練得出來的數字。然後用一個kernel size=3，stride=1的感受野掃描這個區域，這樣就可以得到一個`4*4`的特徵圖了！：

![](https://i.imgur.com/72Va4Sz.png)

我們甚至可以把這個`2*2`的 feature map，每一個畫素點隔開一個空格，空格里的數字填充為0，周圍的padding填充的數字也全都為零，然後再繼續上取樣，得到一個`5*5`的特徵圖，如下所示：

![](https://i.imgur.com/DFDBNyq.png)

## 1*1 卷積
吳恩達教授在講解卷積神經網路的時候，用到了一張十分經典的影像來表示1*1卷積：

![](https://i.imgur.com/zkmaieZ.png)

原本的特徵圖長寬為28，channel為192，我們可以通過這種卷積，使用32個卷積核將`28*28*192`變成一個`28*28*32`的特徵圖。在使用`1*1`卷積時，得到的輸出長款保持不變，channel數量和卷積核的數量相同。可以用抽象的3d立體圖來表示這個過程：

![](https://i.imgur.com/PAGjrVu.png)

因此我們可以通過控制卷積核的數量，將資料進行降維或者升維。增加或者減少channel，但是feature map的長和寬是不會改變的。

### 1 x 1 卷積計算舉例
後期GoogLeNet、ResNet 等經典模型中普遍使用一個像素大小的捲積核作為降低參數複雜度的手段。
從下面的運算可以看到，其實1 x 1 卷積沒有什麼神秘的，其作用就是將輸入矩陣的通道數量縮減後輸出（512 降為32），並保持它在寬度和高度維度上的尺寸（227 x 227）。

![](https://i.imgur.com/VFiOYKD.png)

## 計算 Feature map
Feature map 的輸出大小與 Kernel size、Padding、Strides 有關。通常Padding 又可以分為兩類，分別為：

- Same Padding
    - 在 Stride 為 1  時，會讓輸出Feature map 與輸入圖像維持一樣的尺寸。
- Valid Padding
    - 不會特別去補邊，因此會讓輸出的Feature map尺寸下降，而當遇到卷積無法完整卷積的狀況則會直接捨棄多出來的像素。

卷積過後Feature map尺寸可藉由下方公式計算。如果我們使用 `3*3` filter size， Padding =1, Stride=1，經過計算後就會發現Output=Input。
![](https://i.imgur.com/LdyFBZJ.png)

> Output = (Input-kernel+2*padding)/stride+1

> 最基本就是 image size、kernel size 兩相減再加 1 (padding=0, stride=1 情況下)

- `8*8` image 使用 `3*3` kernel, padding=0, stride=1 => `6*6` feature map
- `5*5` image 使用 `3*3` kernel, padding=0, stride=1 => `3*3` feature map
- `10*10` image 使用 `3*3` kernel, padding=0, stride=1 => `8*8` feature map

![](https://i.imgur.com/cWe6wT6.jpg)
![](https://i.imgur.com/54Wb2Sw.png)

## CNN 架構
基礎CNN圖像辨識模型架構如下圖所示。主要可以拆解為：

> 輸入圖像→[卷積組合→最大池化(Max Pooling)]*n→攤平(Flatten)→全連接層(Fully Connected layers)→分類

其中卷積組合又可以拆分：
> [卷積層→激勵函數(activation function)→Batch Normalization]=卷積組合

![](https://i.imgur.com/6i7ddKs.png) 

## 結論
- 1*1 卷積是拿來控制 channel 維度。
- 上取樣是用來擴增 feature map。
- 做卷積時也可以用 zero padding 讓圖片保持原來大小。
- 2x2 的 Pooling 會讓圖小一半。
- 使用 Relu 函數去掉負值，更能淬煉出物體的形狀

## 筆記
- 理想中影像是一個二維的連續函數
- 一張數位影像是一個 8bit
- 卷積核就像對影像做加權運算
- 透過卷積從低階輸入的能量萃取高階語意的資訊

## Reference
- [卷積神經網路(Convolutional neural network, CNN) ](https://chih-sheng-huang821.medium.com/卷積神經網路-convolutional-neural-network-cnn-卷積運算-池化運算-856330c2b703)
- [深度學習：CNN原理](https://cinnamonaitaiwan.medium.com/深度學習-cnn原理-keras實現-432fd9ea4935)
- [Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
- [著名CNN模型](https://docs.google.com/presentation/d/1AlZkqPa2FylxKsYs2mcz0tpJdgP0Lbno6bYM_fA4UzM/edit#slide=id.p1)

strides=2, pool=2 map 大小減半


rmsle 會將大誤差放小