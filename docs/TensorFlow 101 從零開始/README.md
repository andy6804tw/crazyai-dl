---
title: 'TensorFlow 101 從零開始'
description: '任務導向 TensorFlow/Keras Cookbook，整理 DNN、CNN、RNN、NLP、Transformer、訓練優化與部署流程。'
keywords:
    - TensorFlow
    - Keras
    - Cookbook
    - Deep Learning
tags:
    - TensorFlow 101 從零開始
---

# TensorFlow 101 從零開始：Cookbook 學習地圖

「TensorFlow 101 從零開始」定位為任務導向的 TensorFlow/Keras Cookbook。每篇文章對應一種常見神經網路任務，搭配一份可在 Google Colab 執行的 notebook，目標是讓讀者可以理解流程，也能替換成自己的資料使用。

## 1. 這個系列怎麼使用？

如果你是第一次接觸 TensorFlow，建議從 `0. 系列介紹與學習地圖`、`1. TensorFlow Keras 基礎` 開始。如果你已經知道自己的任務類型，可以直接跳到對應 cookbook：

- 表格資料：看 `3. DNN 表格資料 Cookbook`
- 圖片分類：看 `4. CNN 影像 Cookbook`
- 時間序列：看 `5. 時間序列 Cookbook`
- 文字資料：看 `6. NLP Cookbook`
- Transformer：看 `7. Transformer 入門`
- 模型調不好：看 `8. 訓練優化技巧`
- 模型要交付：看 `9. 模型儲存與部署`

## 2. 系列入口

本系列包含完整的 Cookbook 主線，也保留一篇早期基礎介紹作為補充閱讀：

- [TensorFlow 基礎介紹](./TensorFlow 基礎介紹.md)

如果你想直接從實作任務開始，可以優先閱讀：

- [3.1 DNN Regression](./3.%20DNN%20表格資料%20Cookbook/3.1%20DNN%20Regression.md)
- [3.2 Binary Classification](./3.%20DNN%20表格資料%20Cookbook/3.2%20Binary%20Classification.md)
- [3.3 Multi-class Classification](./3.%20DNN%20表格資料%20Cookbook/3.3%20Multi-class%20Classification.md)

## 3. Cookbook 文章公版

每篇文章都會盡量維持相同結構：

1. 這篇要解決什麼問題？
2. 資料格式長什麼樣子？
3. 載入套件
4. 載入與前處理資料
5. 建立模型
6. 編譯模型
7. 訓練模型
8. 評估模型
9. 預測新資料
10. 如何套用自己的資料？
11. 常見調整方向
12. 小結

## 4. 完整章節

目前 TensorFlow 101 Cookbook 已整理為 0.x 到 9.x：

1. `0. 系列介紹與學習地圖`：環境、路徑與專案流程。
2. `1. TensorFlow Keras 基礎`：TensorFlow、Keras API、Dense layer、compile/fit/evaluate/predict。
3. `2. 資料前處理與 tf.data`：資料切分、標準化、類別編碼、圖片/文字載入與資料管線。
4. `3. DNN 表格資料 Cookbook`：回歸、二元分類、多類別分類、多標籤、不平衡資料、混合特徵與 autoencoder。
5. `4. CNN 影像 Cookbook`：圖片分類、自有圖片資料集、data augmentation、transfer learning、fine-tuning、過擬合與 Grad-CAM。
6. `5. 時間序列 Cookbook`：LSTM、GRU、1D CNN、異常偵測與模型比較。
7. `6. NLP Cookbook`：文字向量化、情緒分類、LSTM、CNN 與 Transformer 文字分類。
8. `7. Transformer 入門`：attention、self-attention、Transformer encoder 與 Vision Transformer。
9. `8. 訓練優化技巧`：callback、learning rate、overfitting、loss/metrics 與 hyperparameter tuning。
10. `9. 模型儲存與部署`：save/load、TensorFlow Lite、TensorFlow.js、FastAPI 與 batch prediction。

你可以依照任務直接跳到對應章節，也可以從 0.x 開始依序建立完整 TensorFlow/Keras 工作流程。
