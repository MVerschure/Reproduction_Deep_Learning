Disclamers for the code: The code inside *GazeTR/* is cloned from https://github.com/yihuacheng/GazeTR. For a manual on how to use this code check their repository. The code in *data_processing_eth.py* is from the authors of a paper discussed below (https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#eth-xgaze)
# Gaze estimation using Transformer

**Group 81:** \
Mink Verschure - 4906020, M.Verschure@student.tudelft.nl\
Luca Fornaro - 5754267, L.Fornaro@student.tudelft.nl\
Yehor Ruban - 5611067, I.Ruban@student.tudelft.nl

## Introduction 

### Explain Assignment

The assignment is focused on reproducing and replicating the code of the existing paper. In our case it is - Gaze estimation using Transformer, which focuses on implementing the “Gaze Transformer” method in solving he problem of estimating the gaze direction by image data.

Reproduction task includes implementation of the existing code of the GazeTR method, provided by the authors of the paper. Reproducing the code implies ensuring that the code is fully functional and generating accurate predictions.

Replication task requires the development of a fully-new implementation of the code, without any usage of the existing version of it. 



### Original Paper


The original paper - **“Gaze estimation using Transformer”**, by Yihua Cheng and Feng Lu, explores the performance of the so-called transformers for gaze estimation in computer vision. 

More specifically, the authors compare the performance of pure transformers versus hybrid transformers (which integrate both transformers and convolutional neural networks) in estimating gaze from images. In the results of the paper it was found that the hybrid transformer significantly outperforms the pure transformer in all evaluation datasets with fewer parameters. The authors also conducted experiments to assess the effectiveness of the hybrid transformer and explore the advantage of self-attention mechanism. The results showed that the hybrid transformer can achieve state-of-the-art performance in all benchmarks with pre-training.


### Why gaze estimation

There are numerous fields where the gaze estimation problem may be useful. These are the few examples of the applications of the gaze estimation:
* <u>Human-Computer Interaction (HCI)</u>: Gaze estimation can be used to develop more natural and intuitive user interfaces for computers and other digital devices. For example, by tracking the user's gaze, a computer could determine where the user is looking on the screen and adjust the display accordingly, or allow the user to interact with the interface using eye movements.
* <u>Virtual and Augmented Reality</u>: Gaze estimation can be used to enhance the immersion and realism of virtual and augmented reality environments by allowing the system to track the user's gaze and adjust the display or interact with virtual objects accordingly. For example, in a virtual game or simulation, the system could adjust the field of view or depth of field based on the user's gaze.
* <u>Medical Applications</u>: Gaze estimation can be used in medical applications, such as diagnosing eye disorders, monitoring eye movements during surgery, or assisting people with disabilities. For example, gaze estimation technology can be used to control a computer or other device through eye movements for people with motor disabilities.
* <u>Market Research</u>: Gaze estimation can be used in market research to track and analyze consumer behavior and preferences. For example, by tracking the gaze of consumers as they view advertisements or products, researchers can determine which elements of the advertisement or product attract the most attention, and use this information to optimize marketing strategies.


### Why is it important to do a reproduction

Papers and codes from the Machine Learning community are very important for various reasons - research to education. Reproducibility, in turn, is a powerful tool for ensuring the quality and validity of these projects findings. Furthermore, reproducibility can help identify errors or issues in the original work, which may have been missed during the initial review process. This can lead to improvements in the methodology, making it more robust and trustworthy.

---

Doing a project reproduction, we will use the given ETH-XGaze dataset to perform the gaze estimation using Gaze Transformer with Hybrid transformer method, described below. However, it is important to note, that the ETH-XGaze dataset was **NOT** used in the original method paper. Therefore, there is no possibility to compare any results with the existing paper. Therefore, we are simply going to provide our own obtained results.

## Method Paper

In this section we will go through the detailes of the method paper used for the project.

As was stated before, authors of the original paper solve the gaze estimaiton problem by applying the transformer neural network architecture, known for its excelent performance in Natural Language Processing (NLP) problems[1]. However, according to the authors, there is no research about using transformers in gaze estimation.

In the project, the authors employed two types of transformers to the task - Pure transformer and Hybrid transformer:

![](https://i.imgur.com/jnIzzBg.png)
<center>Fig. 1: Two types of the transformer - pure transformer and the hybrid one.</center>

### Transformer architecture

Transformers contain encoder and decoder. But in the gaze estimation, the use of transformers is related only to transformer encoders.

The transformer is made up of three main parts: multi-head self-attention (MSA), a two-layer MLP, and layer normalization. Multi-head self-attention allows the model to project queries, keys, and values into multiple subspaces. The output of each subspace is then concatenated and linearly projected to produce the final output. A two-layer MLP is used to introduce non-linearity, and layer normalization ensures stable training and faster convergence. The transformer also uses skip connections. In summary, a one-layer transformer can be expressed as:

$$\overline{x} = MSA(LN(\mathbf{X})) + \mathbf{X}$$
$$x = MLP(LN(\mathbf{\overline{x}})) + \mathbf{\overline{x}}$$

The output x here is of the same dimension as the input X, so it can be stacked for multi-layer transformers.

### Pure transformer

Pure gaze transformer is designed as  in [4]. Main idea behind its application is to divide an original image into N patches, which then serve as a feature vectors. These patches then were mapped into a D-dimensional feature-space using a linear projection. The outcome of that is the N x  D image feature matrix **z**<sub>img</sub>. Also, an extra **z**<sub>token</sub> embedding is added to the image feature matrix. During training, the token uses self-attention to aggregate the features of other patches and outputs gaze representations at the transformer output. To capture the position information of each patch, a position embedding **z**<sub>pos</sub> is created and added to the feature matrix. Therefore the final feature matrix is of the form:

<div style="text-align:center">

$$
\mathbf{z} = [ \mathbf{z}_{token}  ;  \mathbf{z}_{img} ] + \mathbf{z}_{pos}
$$

</div>

This result matrix is then being fed to the transformer,  and MLP is then used to regress an estimated gaze **g**:

<div style="text-align:center">

$$
\mathbf{g} = MLP(Transformer(\mathbf{z})[0, :]) ,
$$

</div>

where we select the first row of the feature matrix as gaze representations.

### Hybrid transformer

However, having a regression task in the end, it is difficult to accurately predict gaze using only local patches, such as a half of an eye image. These local patches do not provide enough information about the overall context of the image, and may result in inaccurate predictions.

To address this issue, a CNN is used to extract local features from the eye images, because CNN can capture low-level visual patterns and local features in the images, such as edges, shapes, and textures. The output feature maps from the CNN contain information about the local regions of the input image.

The transformer is then used to capture global relations and dependencies between these local features. It can learn to model the complex relationships between different regions of the input image, and can effectively aggregate information from the local feature maps to predict the gaze point. Therefore, given a face image, CNN is being applied to acquire the feature map, that is then reshaped into a 2D patch. The following steps are similar to the pure transformer application principle.

## Dataset

In this paper reproduction project we use the **ETH-XGaze** dataset. This is a dataset for gaze estimation that was created by the Computer Vision Lab at ETH Zurich. The dataset contains over 1 million images of gazes, along with corresponding annotations for gaze direction. ETH-XGaze was collected from 110 participants of with a custom hardware setup including 18 digital SLR cameras and adjustable illumination conditions, and a calibrated system to record ground truth gaze targets. The dataset samples large variations in head poses, up to the limit of where both eyes are still visible (maximum ±70 degrees from directly facing the camera) as well as comprehensive gaze directions (maximum ±50 degreesin the head coordinate system)


![reference link](https://i.imgur.com/14M4tOw.jpg)

<center>Fig. 2: Data collection device as well as some image examples</center>


## Reproduction
To replicate the original papers code<sup>1</sup>, multiple steps were taken: The data was preprocessed, some trouble shooting was done for training and evaluating the model and finally the results were visualised.

### Working with the HPC cluster (Hyper Performance Computing)

In order to carry out this reproducibility project, we needed to use the HPC Cluster (Hyper Performance Computing) to access both the dataset and to run the codes, given the large size of the dataset. So, we gained access to the HPC cluster with linux-bastion, set up a new pytorch environment, and used Slurm Workload Manager job scripts to operate on the cluster.

### Data preprocessing
The data preparation consisted of converting it into a proper format which the model can use as a input: The dataset provides a .h5 file for each subject containing many images and their labels. These .h5 files are unpacked into images and files containing all the labels, and sorted into training and testing sets using the preprocessing code provided on the GazeHub website<sup>2,3</sup>.

### Model training
We utilized the code provided by the authors of the original paper to train our model. Due to the large size of the original dataset, we decided to use a reduced dataset consisting of 35 people for the training set and 15 people for the test set. This allowed us to experiment with different parameters and ensured a reasonable amount of training time. Our training process utilized a batch size of 512, a learning rate of 0.0005, and 50 epochs. We implemented a weight decay of 0.5 every 10 epochs. Following the authors steps, we also trained an already pre-trained model using our own dataset. The figures below depict the training losses for each epoch of both models.

<center><img src="https://i.imgur.com/XYNF7QR.png"  width="500" height="280"></center>
<center>Fig. 3: Training loss on reduced training set, no pre-trained model </center>
<br>

<center><img src="https://i.imgur.com/i5Onoo0.png"  width="500" height="280"></center>
<center>Fig. 4: Training loss on reduced training set, pre-trained model </center>
<br>

It can be immediately observed that the curve decreases until it converges to different values, which are still very low. In fact in the not pre-trained model the final train loss is around 0.015, while the one for the pre-trained model is around 0.007.
The training time lasted around 11 hours for both models. 

### Model testing and trouble shooting
After training the model, we tested it on a part of the test set provided directly from the original data (we used 15 people). The results initially were not good at all. In fact, pictures below depict the test loss and accuracy, calculated as the angle, in degrees, between the true label and the predicted label of the not pre-trained model. All result visualization was obtained using TensorBoard.

<center><img src="https://i.imgur.com/UrZoF0C.png"  width="500" height="280"></center>
<center>Fig. 5: Test loss on reduced test set, not pre-trained model </center>
<br>

<center><img src="https://i.imgur.com/MDqMgvg.png"  width="500" height="280"></center>
<center>Fig. 6: Average error on reduced test set, not pre-trained model </center>
<br>

These results report that the model was not able to generalize well to the test data: in fact, the test loss was very high even after 50th epoch (about 0.4). Thus, the error value was about 27.5 degrees, which is a rather large error of gaze estimation.

This led us to investigate the reason for such a large prediction error, leading us to study losses in both training and testing, although the time required for training brought difficulties. Given the previous plots, our first conclusion was that the model overfitted the data: in fact, the train loss is very low, while the test loss is high. However, considering the exellent performance obtained by the authors, that assumption was insufficient. 

In the end, we noticed that the problem was in the dataset. The test data labels contained only the head pose direction. That was leading to such a poor performance, as the model basically predicted the head position.

To solve this problem we then took a part of the original training data with correct labels, which was not involved in pre-training the model, and used it as a test set. 

### Final results
We finally reused the previously obtained models (the not pre-trained model and the pre-trained one) to get the final results on the properly labeled test sets. Below are the test losses of the two models on a properly labeled test set.

<center><img src="https://i.imgur.com/ED2CUkm.png"  width="500" height="280"></center>
<center>Fig. 7: Test loss on reduced test set correctly labeled, not pre-trained model </center>
<br>

<center><img src="https://i.imgur.com/hUTReUp.png"  width="500" height="280"></center>
<center>Fig. 8: Test loss on reduced test set correctly labeled, pre-trained model </center>
<br>

As can be seen, the results now are significantly better in comparison with the previous ones. This is also observable in the accuracy value:

<center><img src="https://i.imgur.com/GgwV3AG.png"  width="500" height="280"></center>
<center>Fig. 9: Average error on reduced test set correctly labeled, not pre-trained model </center>
<br>

<center><img src="https://i.imgur.com/ZK2jxbk.png"  width="500" height="280"></center>
<center>Fig. 10: Average error on reduced test set correctly labeled, pre-trained model </center>
<br>

Now the predictions are very accurate: in case of the non-pretrained model, the average error is only 6 degrees, while for the pretrained model the error is around 1.87 degrees. These are very good results, especially considering that we did not use the whole dataset. This shows that the model is effetively accurate and can correctly predict gaze direction.

The result of the gaze estimation, performed by the model can be seen on the Fig. 11 - here gaze directions are indicatted with arrows. Green arrow represents the gaze as predicted by the model, while red arrow represents the gaze as labeled in the dataset:


![image alt <](https://i.imgur.com/qOOMihZ.jpg)
![image alt ><](https://i.imgur.com/bWW6lSz.jpg)
![image alt >](https://i.imgur.com/9hxBlrr.jpg)
<center>Fig. 11: Visualized results of the model.</center>
<br>

As you could have noticed from the Fig.11, the end results show that the model estimates gaze with a remarkable accuracy, resulting in the gaze direction prediction that is visually almost indistinguishable from the actual label.

## Sources
1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. doi: https://doi.org/10.48550/arXiv.1706.03762
2. Y. Cheng, H. Wang, Y. Bao, and F. Lu, “Appearance-based gaze estimation with deep learning: A review and benchmark,” arXiv preprint arXiv:2104.12668, 2021
3. X. Zhang, S. Park, T. Beeler, D. Bradley, S. Tang, and O. Hilliges, “Eth-xgaze: A large scale dataset for gaze estimation under extreme head pose and gaze variation,” in The European Conference on Computer Vision (ECCV), 2020.
4. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.


---




