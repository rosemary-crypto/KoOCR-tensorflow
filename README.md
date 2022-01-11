
# KoOCR-tensorflow (Korean README)

An open source Korean OCR engine based on Tensorflow deep learning.
Open-source Korean OCR engine based on Tensorflow, deep-learning.

![](files/hangeul_image.png)

## Overview
- Compared to the excellent OCR recognition performance of similar languages ​​such as Chinese and Japanese, active research on Korean recognition has not been conducted.
- There were not many high-performance Korean OCR projects and libraries that could be used easily. 
- The learning method and model architecture used in Chinese recognition (HCCR) were applied to Korean recognition and the performance was compared.
- Due to the special structure of Hangeul, a multi-output model that predicts the initial, middle, and final consonants separately was constructed. 


##  Method and Plans


- [x]  DirectMap: Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark
- [x]  Fire-module based model, GWAP: Building Efficient CNN Architecture for Offline Handwritten Chinese Character Recognition
- [x] High-performance network architecture, CAM, GAP/GWAP/GWOAP: A High-Performance CNN Method for Offline Handwritten Chinese Character Recognition and Visualization
- [ ] Hybrid learning loss: Improving Discrimination Ability of Convolutional Neural Networks by Hybrid Learning
- [ ] Adaptive Drop Weight, GSLRE: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition
- [x] Adversarial Feature Learning: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data
- [ ] DenseRAN style model: DenseRAN for Offline Handwritten Chinese Character Recognition
- [x] Iterative Refinemet: Improving Offline Handwritten Chinese Character Recognition by Iterative Refinement

Plan to write a thesis that implements the method suggested in the above papers and evaluates the degree of performance improvemen of Hangul recognition.

## Model Performance

Pretrained weights are stored under the `pretrained/해당모델` folder. Additional training results for each model are stored in `pretrained/해당모델_evaluation`. Confusion matrix for each class, Grad-CAM results, etc. are saved. The table below shows the comparison results of each model and learning method. 

model type              | Typography Accuracy| Handwriting Accuracy | Inference time (ms/image) | Model size (mb)
----------------------- | ------------------ | -------------------- | ------------------------- | --------------
plain_melnyk_complete   |99.94%              |97.94%                |                           |57.1

### plain_melnyk_complete
A baseline model of High-performance network architecture(melnyk network). It is the result of 1 epoch with a learning rate of `0.001` and 2 epochs with a learning rate of `0.00003`. [Adabound optimizer](https://arxiv.org/abs/1902.09843) was used. `fc_link` refers to a method of connecting the output feature map of a convolutional neural network to a fully connected layer, and GWAP, which is an improved GAP (Gradient Average Pooling) proposed in this paper, is used.
```
!python train.py --learning_rate=0.00003 --optimizer=adabound --image_size=96 --weights=./logs/weights/ --fc_link=GWAP --batch_size=128 --epochs=2 --patch_size=20 
```


## Use Project

### load_data.py
```
!python load_data.py --sevenzip=true
```
Download dataset uploaded to Google Drive and get `./data`, `. /val_data` is saved. The dataset consists of 300 .pickle patches, and they are named `handwritten_*.pickle`, `printed_*.pickle`, and `clova_*.pickle` according to the characteristics of the data. It represents the image of handwriting, printed type, and handwritten font, respectively.

The `sevenzip` variable indicates whether to receive the compressed data as a .7z file or a .zip file. A value of True speeds up download and compression. 

### crawl_data.py

```
!python crawl_data.py   --AIHub=true 
			--clova=true
			--image_size=96
			--x_offset=8
			--y_offset=8
			--char_size=80   
```
Dataset is crawled and downloaded. It has the same role as load_data.py. The `x_offset`, `y_offset`, and `char_size` variables specify the offset of the position and the size of the character when the font is drawn on the image. The table below shows the variable setting values according to the image size used in the experiment. 

 
image size | x_offset | y_offset | char_size
---------- | -------- | -------- | ---------
64         |         5|         5|50
96         |8         |8         |80
128        |14        |10        |100
256        |50        |10        |200


### model. py
```
import model
OCR_model=model.KoOCR(weights='C:\\...', split_components=True, ...)

OCR_model.model.summary()
OCR_model.train(epochs=10, lr=0.01, ...)

pred=OCR_model.predict(image, n=5)
```
As a module defining model, `KoOCR` class is defined. **The method `KoOCR.predict` used for inference receives an image or image arrangement and returns the top-n number of Hangul characters that are most likely.** The method used for additional training of the model is `KoOCR.train' `, and learning is carried out based on the received hyperparameter.


### train. py
```
!python train.py--split_components=true 
		--network=melnyk
		--image_size=96
		--direct_map=true
		--epochs=10
					...
```
Although it is a Python module that conducts learning, it only defines the model and calls `KoOCR.train`, but there is no difference from directly importing `model.py` and training. All information about learning results and processes is stored in `./logs`, and weights are stored in `./logs/weights.h5` at every epoch. 

 
### evaluate. py
```
python evaluate.py	--weights='./logs/weights.h5'
			--accuracy=true
			--confusion_matrix=true
			--class_activation=true
```
The model is analyzed in three ways: accuracy, confusion matrix, and CAM. You can deselect each method or set detailed parameters for each method such as top-n accuracy.  
