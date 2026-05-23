---
title: '神經網路導論與實作'
description: '從手寫數字資料觀察、SVM baseline、SVHN 真實影像資料，到 TensorFlow DNN 與 CNN 的入門實作。'
keywords:
    - neural network
    - TensorFlow
    - DNN
    - CNN
    - SVM
    - SVHN
tags:
    - 神經網路導論與實作
---

# 神經網路導論與實作

這個章節以「從資料觀察到模型比較」作為學習主線，帶領讀者從簡單的手寫數字影像開始，逐步理解傳統機器學習與深度學習在影像分類任務上的差異。

![](./image/img-nn-workshop-cover.png)

本系列不是從公式開始背起，而是先讓模型真的跑起來。讀者會先觀察圖片資料如何以陣列形式儲存在電腦中，再使用 SVM 建立第一個分類 baseline，接著切換到更接近真實世界的 SVHN 街景門牌數字資料集，觀察傳統機器學習在複雜影像上的限制，最後使用 TensorFlow 建立 Dense DNN 與 CNN 模型。

## 1. 學習目標

完成本章節後，你將能夠：

1. 了解灰階影像與彩色影像在電腦中如何表示。
2. 使用資料視覺化觀察手寫數字資料集的特徵。
3. 使用 SVM 建立傳統機器學習分類模型。
4. 說明為什麼影像資料常需要正規化或標準化。
5. 使用 TensorFlow/Keras 建立 Dense DNN 模型。
6. 透過神經網路優化技巧改善模型表現。
7. 理解 CNN 為什麼是影像辨識任務中的重要模型架構。
8. 比較不同模型在不同資料複雜度下的表現。

## 2. 課程主線

本系列的設計順序如下：

| 章節 | 主題 | 對應 Notebook |
|---|---|---|
| 1 | 圖片資料與手寫數字辨識 | `01_digits_data_visualization.ipynb` |
| 2 | SVM 手寫數字分類 baseline | `02_digits_svm_baseline.ipynb` |
| 3 | SVHN 真實世界數字影像與 SVM 限制 | `03_svhn_svm_baseline.ipynb` |
| 4 | TensorFlow Dense DNN 入門 | `04_svhn_dnn_tensorflow.ipynb` |
| 5 | DNN 優化技巧與過擬合 | `05_svhn_dnn_optimization.ipynb` |
| 6 | CNN 影像辨識入門 | `06_svhn_cnn_tensorflow.ipynb` |
| 7 | 模型比較與學習總結 | 綜合比較 |

## 3. 為什麼先做 SVM，再做 DNN？

傳統機器學習方法，例如 SVM，對於特徵清楚、資料規模適中的任務仍然非常有用。在簡單的手寫數字資料上，SVM 可以取得相當好的效果。這代表傳統機器學習並不是「過時」的技術，而是適合的任務不同。

但當資料變得更貼近真實世界，例如 SVHN 街景門牌數字，圖片會出現背景、顏色、光線、角度與周圍雜訊。這時候若只把圖片攤平成一長串 raw pixels，再交給傳統機器學習模型，模型很難自動理解局部紋理與空間結構。

深度學習的價值就在這裡：模型不只是在已經整理好的特徵上分類，也能從資料中學出更有用的表示方式。

## 4. Colab 實作方式

每篇實作文章上方都會附上 Colab badge。點擊後即可在 Google Colab 中開啟 notebook，不需要在本機安裝 TensorFlow 環境。

!!! note

    建議依照章節順序完成實作。前兩份 notebook 使用較簡單的手寫數字資料，重點是建立資料與分類流程的直覺；後四份 notebook 會切換到 SVHN 資料集，觀察模型在真實影像任務中的表現差異。

## 5. 模型表現總覽

本系列會逐步得到以下觀察：

| 階段 | 模型 | Train Accuracy | Test Accuracy |
|---|---:|---:|---:|
| 簡單手寫數字 | SVM RBF | 高準確率 | 高準確率 |
| SVHN | SVM RBF raw pixels | 0.6947 | 0.5170 |
| SVHN | Simple Dense DNN | 0.7476 | 0.6540 |
| SVHN | Optimized Dense DNN | 0.8501 | 0.6973 |
| SVHN | Small CNN | 0.9814 | 0.9263 |

!!! info

    簡單手寫數字資料集與 SVHN 是不同資料集，不能直接當成公平模型排行。這裡的重點是觀察：當資料從乾淨、簡單走向真實、複雜時，不同模型的能力差異會逐漸浮現。
