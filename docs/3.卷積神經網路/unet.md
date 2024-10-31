# 圖像分割經典網路 U-Net
影像分割另外一個具有代表性的 CNN 經典模型是 U-net，他是被用來處理醫療影像並發表於 MICCAI 2015。U-Net 與 FCN 同樣都是沒有全連接層只有捲積層，同樣是逐層融合不同尺度特徵圖，但U-Net更加像一個非常規整的Encoder-Decoder結構。並且都是採用 deconvolution 將低解析度影像還原高解析度，但是整體的 Upsampling 過程兩者是不太一樣的。

![](https://i.imgur.com/irPxc5F.png)

## U-Net 架構解析
因為網路架構長得很像 U 型因此稱作 U-net，又因為實作時會拆分 Encoder-Decoder 因此我們又稱它是個 AutoEncoder 的結構。論文中提出，左半的編碼部分叫 Contracting Path，用於萃取重要訊息。右邊的解碼部分叫 Expanding Path，用於精確定位還原解析度。

整體架構如下圖所示當我們輸入一張 `572*572` 的影像，藍線箭頭代表進行了 `3*3` 的卷積並經過 relu 函數，所以一開始經過兩層卷機後得到 64 張的 `568*568` 特徵圖。而紅色的箭頭代表經過一個 `2*2` 的 poling，此步驟會將影像大小減半。在左半邊總共進行了四組 downsampling 
操作後得到的最低解析度為 `32*32`。接下來一樣會有四組 Upsampling 還原到高解析度，值得一提的是這裡的 Upsampling 過程與 FCN 不一樣。雖然都是採用 deconvolution 但是處理 feature maps 的方法不一樣。首先將 `28*28` 影像進行反卷積得到 `56*56`，可是我們發現他的跳耀連接左邊是 `64*64` 大小。此時的做法會將左邊 `64*64` 的影像取中間符合右邊 56*56 的大小進下行拼接變成 1024 維。此動作一樣在 Upsampling 進行四組最後得到 `392*392` 的大小。接著再做兩組 3*3 的卷積最後再透過一個 `1*1` 卷積得到 `388*388` 通道為 2 的影像分別是細胞與非細胞的機率圖。

![](https://i.imgur.com/3RUi691.png)

一個正規的 U-net 架構是由左邊 contracting path 和右邊 expanding path 組合而成。其中 contracting path 是由卷積與最大池化所組成，因此解析度會逐漸降低。而 expanding path 是由反卷積(論文稱 Up-conv) 與左邊的高解析度影像 channel-wise 組合。


## FCN 與 U-net 差異
因此我們這裡可以做一個統整比較 FCN 與 U-net 的差異。FCN 在 Upsampling 是透過 element-wise 與前一層的特徵圖相加。而 U-net 是採用左邊高解析度資訊 channel-wise concatenation 組合而成。

FCN中使用的是逐點相加的方式 add()，而U-Net中是按 channel 維度拼接在一起 concat()，直接將通道數翻倍，形成更厚的特徵圖。

## 影像前處理： Overlap-Tile Strategy 方法
另外在 U-net 架構中卷積層並無使用 padding，因此每過一層卷積特徵圖的大小都會逐漸縮小。輸入是 572×572，最終輸出是 388×388。為了控制解析度和沒有 padding 所造程的影想論文中提出 Overlap-Tile Strategy 方法。假設我要預測圖中黃色框框的圖像分割內容，就以黃色為中心重新裁切一個較大的藍色框框的影像作為輸入，經過每一層不斷的卷積最後出出的結果剛好是黃色的區域。

![](https://i.imgur.com/mGM5MNS.png)

> 在實作過程中卷積可以採用 padding=same 控制特徵圖大小。並採用 pooling 機制將特徵圖縮小。這樣就可以解決輸入輸出影像大小不一的問題。

## 實驗結果
論文裡採用兩個公開醫療影像資料集進行實驗。我們可以發現 U-net 在 IOU 的評估上都獲得相當好的成績。

![](https://i.imgur.com/uOlNDMe.png)

## QA
- 關於 CNN 若每一層卷積的 kernel size 都等於原圖 H*W 是不是就等同於全連接層網路？

- CNN中的convolution單元每次只關注鄰域kernel size 的區域，就算後期感受野越來越大，終究還是局部區域的運算。這一點該如何解決？採用注意力機制可以解決這個問題嗎

- FCN 有使用到 Unpooling 技巧嗎還是只有採用反卷機進行 upsampled？FCN 論文圖中架構是基於 VGG16 嗎？

- U-Net與FCN幾乎都是同一時間出現。想問老師U-Net是受到FCN啟發而出現的架構，還是兩者剛好都是在同一年受到先前的反卷積啟發呢。

- Mask R-CNN 一開始的 Feature extractor 採用 RestNet 是 pretrain 好的模型嗎？真實在訓練時要把 trainable 打開或是只開後面幾層呢？

- RoIAlign 簡報中的例子假設在 2*2 情況下一個 cell 有四個透過雙線性內差法得到的點。接著再透過 max pooling 保留一個點嗎？想請教 RoIAlign 最後如何採用 max pooling 保留具有代表性的點，是否能舉簡單例子。

- FCN 跟 U-net 在 real time 上的效能比較。

- Mask R-CNN 效能大約 5fps 請問是在什麼規格下呢？像是影像大小、GPU 幾顆。
