
# KoOCR-tensorflow (Korean README)

Tensorflow 딥러닝 기반의 오픈 소스 한글 OCR 엔진.
An open source Korean OCR engine based on Tensorflow deep learning.
Open-source Korean OCR engine based on Tensorflow, deep-learning.

![](files/hangeul_image.png)

## 개요
- 중국어, 일본어 등 유사한 언어의 뛰어난 OCR 인식 성능에 비해 한글 인식에 대해서는 활발한 연구가 이루어지 않았다.
- 쉽게 사용 가능한 고성능의 한글 OCR 프로젝트, 라이브러리가 많지 않았다. 
- 중국어 인식(HCCR)등에 사용된 학습 방법, Model Architecture를 한글 인식에 적용하고 성능을 비교하였다.
- 한글의 특수한 구조에 기인해 초성, 중성, 종성을 각각 따로 예측하는 Multi-output 모델을 구성했다. 
## Overview
- Compared to the excellent OCR recognition performance of similar languages ​​such as Chinese and Japanese, active research on Korean recognition has not been conducted.
- There were not many high-performance Korean OCR projects and libraries that could be used easily. 
- The learning method and model architecture used in Chinese recognition (HCCR) were applied to Korean recognition and the performance was compared.
- Due to the special structure of Hangeul, a multi-output model that predicts the initial, middle, and final consonants separately was constructed. 


##  Method and Plans
@@ -25,32 +25,32 @@ Open-source Korean OCR engine based on Tensorflow, deep-learning.
- [ ] DenseRAN style model: DenseRAN for Offline Handwritten Chinese Character Recognition
- [x] Iterative Refinemet: Improving Offline Handwritten Chinese Character Recognition by Iterative Refinement

위 논문들에서 제시하는 method를 구현하고 한글 인식에 대한 성능 개선의 정도를 평가하는 논문을 작성할 계획이다. 
Plan to write a thesis that implements the method suggested in the above papers and evaluates the degree of performance improvemen of Hangul recognition.

## 모델 성능
## Model Performance

pretrained weights는 `pretrained/해당모델` 폴더 아래 저장되어 있다. 각 모델에 대한 추가적인 학습결과는 `pretrained/해당모델_evaluation`에 저장되어 있다. Class별 confusion matrix, Grad-CAM 결과 등이 저장되어 있다. 아래 표는 각 모델과 학습 방법의 비교 결과이다. 
Pretrained weights are stored under the `pretrained/해당모델` folder. Additional training results for each model are stored in `pretrained/해당모델_evaluation`. Confusion matrix for each class, Grad-CAM results, etc. are saved. The table below shows the comparison results of each model and learning method. 

model type              | 인쇄체 정확도 | 손글씨 정확도 | 추론 시간(ms/image) | 모델 크기(mb)
----------------------- | ------------ | ------------- | ------------------- | --------------
plain_melnyk_complete   |99.94%        |97.94%         |                     |57.1
model type              | Typography Accuracy| Handwriting Accuracy | Inference time (ms/image) | Model size (mb)
----------------------- | ------------------ | -------------------- | ------------------------- | --------------
plain_melnyk_complete   |99.94%              |97.94%                |                           |57.1

### plain_melnyk_complete
High-performance network architecture(melnyk network) 의 baseline 모델. `0.001`의 학습률로 1 에포크, `0.00003`의 학습률로 2 에포크 학습한 결과이다. [Adabound optimizer](https://arxiv.org/abs/1902.09843)를 사용하였다. `fc_link`는 합성곱 신경망의 output feature map을 fully connected layer로 연결하는 방법을 의미하는데, 본 논문에서 제시한 GAP(Gradient Average Pooling)을 개선한 GWAP를 사용하였다. 
A baseline model of High-performance network architecture(melnyk network). It is the result of 1 epoch with a learning rate of `0.001` and 2 epochs with a learning rate of `0.00003`. [Adabound optimizer](https://arxiv.org/abs/1902.09843) was used. `fc_link` refers to a method of connecting the output feature map of a convolutional neural network to a fully connected layer, and GWAP, which is an improved GAP (Gradient Average Pooling) proposed in this paper, is used.
```
!python train.py --learning_rate=0.00003 --optimizer=adabound --image_size=96 --weights=./logs/weights/ --fc_link=GWAP --batch_size=128 --epochs=2 --patch_size=20 
```


## 프로젝트 사용
## Use Project

### load_data.py
```
!python load_data.py --sevenzip=true
```
Google Drive에 업로드된 데이터셋을 다운로드 받아 `./data`, `./val_data` 에 저장한다. 데이터셋은 300여개의 .pickle 패치로 이루어져 있고, 데이터의 특성에 따라 `handwritten_*.pickle`, `printed_*.pickle`, `clova_*.pickle` 으로 이름지어져있다. 각각 손글씨, 인쇄체, 손글씨 폰트의 이미지를 나타낸다. 
Download dataset uploaded to Google Drive and get `./data`, `. /val_data` is saved. The dataset consists of 300 .pickle patches, and they are named `handwritten_*.pickle`, `printed_*.pickle`, and `clova_*.pickle` according to the characteristics of the data. It represents the image of handwriting, printed type, and handwritten font, respectively.

`sevenzip` 변수는 압축된 데이터를 .7z 파일로 받을지 .zip 파일로 받을지를 나타낸다. True 값이 다운로드와 압축 속도가 빠르다. 
The `sevenzip` variable indicates whether to receive the compressed data as a .7z file or a .zip file. A value of True speeds up download and compression. 

### crawl_data.py

@@ -62,7 +62,8 @@ Google Drive에 업로드된 데이터셋을 다운로드 받아 `./data`, `./va
			--y_offset=8
			--char_size=80   
```
데이터셋을 크롤링해서 다운로드받는다. load_data.py와 같은 역할을 한다. `x_offset`, `y_offset`, `char_size` 변수는 폰트를 이미지에 그릴 때 위치의 offset과 문자의 크기를 지정한다. 아래 표는 실험에서 사용한 이미지 크기에 따른 변수 설정값이다. 
Dataset is crawled and downloaded. It has the same role as load_data.py. The `x_offset`, `y_offset`, and `char_size` variables specify the offset of the position and the size of the character when the font is drawn on the image. The table below shows the variable setting values according to the image size used in the experiment. 


image size | x_offset | y_offset | char_size
---------- | -------- | -------- | ---------
@@ -82,7 +83,8 @@ OCR_model.train(epochs=10, lr=0.01, ...)
pred=OCR_model.predict(image, n=5)
```
모델을 정의하는 모듈으로 `KoOCR` class가 정의되어 있다. **추론에 사용되는 메소드 `KoOCR.predict`는 이미지 혹은 이미지의 배치를 입력받아 가능성이 가장 높은 top-n 개의 한글 글자를 반환한다.** 모델의 추가적인 학습에 사용되는 메소드는 `KoOCR.train`으로, 입력받은 Hyperparameter를 바탕으로 학습을 진행한다. 
As a module defining model, `KoOCR` class is defined. **The method `KoOCR.predict` used for inference receives an image or image arrangement and returns the top-n number of Hangul characters that are most likely.** The method used for additional training of the model is `KoOCR.train' `, and learning is carried out based on the received hyperparameter.


### train. py
```
@@ -93,7 +95,8 @@ pred=OCR_model.predict(image, n=5)
		--epochs=10
					...
```
 학습을 진행하는 파이썬 모듈인지만, 모델을 정의하고 `KoOCR.train`을 호출하는 역할을 할 뿐, 직접 `model.py`를 import 하고 훈련하는 것과 차이가 없다. 학습결과와 과정에 대한 모든 정보는 `./logs`에 저장되고, 가중치는 매 에포크마다 `./logs/weights.h5`에 저장된다. 
Although it is a Python module that conducts learning, it only defines the model and calls `KoOCR.train`, but there is no difference from directly importing `model.py` and training. All information about learning results and processes is stored in `./logs`, and weights are stored in `./logs/weights.h5` at every epoch. 


### evaluate. py
```
@@ -102,4 +105,4 @@ python evaluate.py	--weights='./logs/weights.h5'
			--confusion_matrix=true
			--class_activation=true
```
모델을 정확도, confusion matrix, CAM 3가지 방법으로 분석한다. 각 방법을 선택 해제하거나 top-n 정확도 등 각 방법의 세부적인 parameter 또한 설정할 수 있다. 
The model is analyzed in three ways: accuracy, confusion matrix, and CAM. You can deselect each method or set detailed parameters for each method such as top-n accuracy.  
