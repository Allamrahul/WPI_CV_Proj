# Monocular Dense Depth Map estiamtion using CNN

We will be using this repository to document our code and our work on RBE 549 term project at WPI, MA.

# Project Goal

The goal is to experiment using deep learning networks to estimate depth from a single 2D image i.e Monocular depth estimation.

# Dataset

We use the NYU Depth V2 dataset to train out CNN architecture. It is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinetic. Further, it consists of scenes like bedroom, living room, bathroom, kitchen , bookstore , café etc.​ We use a subset of 10,000 paired images of RGB and Depth for our model training and validation.

**Official Dataset** : https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

**Preprocessed NYU Depth V2 dataset in HDF5 format**: http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz

# Problem Formulation

Given a single 3 channel RGB image as input, predict a dense depth map for each pixel. ​

Given a training set T = {(I<sub>i</sub>, D<sub>i</sub>)}<sup>M</sup>, where i= 1 to M, I<sub>i</sub> ∈ I and D<sub>i</sub> ∈ D, 

the task is to learn a non-linear mapping Φ: I → D, where I is 2D 3 channel RGB Image and D is Depth map.​

# DATA PRE-PROCESSING

Each image pair in the training data is encoded as a.h5 file, which contains both an RGB image and its corresponding depth map of size 640 x 480. We convert this data into respective NumPy arrays, apply a group of image transformation on both RGB and depth maps, where  we perform operation such  as  resize, center crop to reduce the spatial size of the RGB images to 224 x 224 x 3 and depth maps to 224 x 224 x 1. Post this, the images are converted into tensors and are fed to the  architecture.

# Methodology
 
 In this section, we describe the network architecture followed by justification for the choice behind the network.

![archi](https://user-images.githubusercontent.com/18104407/146618637-16aaa2a5-666b-45dd-b208-654d986fe3fc.png)

Our network is a fully convolutional encoder-decoder architecture. The encoder performs depth-wise separable convolutions and extracts features from the input image. The output of the MobileNet encoder is fed to the decoder, where the feature maps  are upsampled  at  each  stage  and  finally  give  the output  depth  map. The feature  maps  extracted  at  each layer of  the encoder  are fed  to the corresponding  decoder  of  appropriate  feature  map  size using  skip  connections. These feature maps from the encoders enable the decoder to get spatial information lost during the propagation and help in recovering fine-grained details. Finally, we use 1x1  convolution at the end to get the Dense Depth Map. The light-weight and low latency nature of the architecture helped us achieve 30fps real-time dense depth  map  playback, which can  be  used  in  applications such as mobile robotics.

# Encoder Network

To get low latency,we use a state-of-the-art efficient network, MobileNet (Andrew G. Howard, 2017), as our encoder of choice. MobileNet makes use of depth-wise decomposition, which  are  more  efficient  than  normal  convolutions. Below, we present the particulars of the encoder architecture, which captures the filter shape, stride and input size of each layer.

![det](https://user-images.githubusercontent.com/18104407/146618492-0663402d-07d5-4c84-9a8b-79d82c85d148.png)

In the figures above, a 224 x 224 x 3 tensor is being fed to the network, which gets processed by the network. As we the process the tensor through the layers of the encoder, the spatial size of the image is decreased while the representative power/depth/number of channels of feature map increases. Finally, after we process the tensor through all encoder layers, we reach the bottleneck, represented in red, where we get a 7 x 7 x 1024 feature map. The network performs average pooling before it feeds the output to the decoder architecture.

# Depthwise Separable Convolutions

Depthwise separable convolutions (Chollet, 2017) help the encoder encode the data faster than traditional convolutions. In the figure below, we present a pictorial representation to compare standard convolution and depthwise convolution.

![dw](https://user-images.githubusercontent.com/18104407/146618905-1aa26fbc-0246-40ae-8262-4375e4d4dc1c.png)

On the left, you can see the whole filter being convolved with the 3 channel RGB input to compute the convolved output. However, on the right, each channel of the 3-channel filter is being convolved individually with each channel of the input 3 channel image to get a 3 convolved feature maps, only to apply a point-wise convolution to compute the final feature map. Both convolutions result in the same answer but depthwise separable convolution is 100 times faster than traditional convolution.

# Decoder Network

The objective of the decoder is and upsample the output of the encoder to form a dense prediction. An aspect of the decoder is the up-sample operation used with
strided convolution/ fractional convolution/deconvolution. Convolution is then by nearest-neighbor interpolation that doubles the number of spatial resolutions. This results in a fast decoder which outputs the decoder size. The decoder consists of five blocks each of the module having a strided convolution, upsampling followed by Batch Normalization and Relu activation. The decoder increases the spatial resolution by increasing width and height. The input to the decoder from the bottleneck layer is 7x7x1024 which is then upsampled and convolved using strided convolution. This is upsampled through modules up to 224x224x32 on which we finally apply 1x1 convolution to get 224x224 x1 depth map output.

# Skip Connections

Encoder networks like above contain many layers to gradually reduce the spatial resolution, where we extract low level features to complex features as we go
deeper in the network. Skip connections here are long skip connections which allows us to get localization features which are lost to the convolutions and down sampling through various layers. Skip connections like these were previously used in architectures such as U-Net (Olaf Ronneberger, 2015), showing that they can be beneficial in networks producing dense outputs like ours. We include skip connections from the MobileNet encoder to the outputs of the middle three layers in the decoder.

# Justifications
In small robotics platforms, multiple programs (such as localization, mapping, control, and other potentially other perception tasks) all run in parallel. Each
program demands a certain amount of computational resources, and consequently, the CPU is not dedicated to the depth estimation task. Hence, our goal was to implement a lightweight depth estimation network which can be used in applications with computational restraints and resource limitations. Hence, we used MobileNet, which is a fast encoder due to its inherent depthwise separable convolutions. The network has 3 million trainable parameters, which is 13 times
less in comparison to other depth estimation architectures (Wonka). Despite this, it performs pretty well while not compromising much on performance.

# Training
Below, we show the training trials and plot the RMSE loss against epochs.

![training](https://user-images.githubusercontent.com/18104407/146619235-75fe9f57-96b4-4ac2-bff4-21fa82111071.png)

We train the network with multiple hyper-parameters such as learning rate, loss type and batch size. We try two different losses such as L1 loss (mean absolute error) and L2 loss (mean square error), which penalizes the loss highly. We experiment using different batch sizes to see if our model generalizes better. We also trained our model with batch size 256 to gauge any shift in performance. On the right, we plot the root mean square error (RMSE) against the number of epochs. We see that the RMSE error would not decrease beyond 0.55m and this is the best loss the model/architecture can achieve. Additionally, even though the model was trained for 70 epochs, the model’s RMSE reached 0.55m by the 10th epoch. Post that, the loss saturates and will not decrease any further.


# Validation
Firstly, we talk about the validation metrics we used to evaluate the performance of our model and present our metrics for each of our training trials. Further, we present the dense depth maps generated by the architecture and discuss our inferences and deductions. We present the root mean square error (RMSE) values and δ1 accuracy (the percentage of predicted pixels where the relative error is within 25%) for each of our training trails.

![rmse](https://user-images.githubusercontent.com/18104407/146619366-4e448712-2840-4833-ad41-8de53c4a5473.png)

Above, our best model achieved 0.544 RMSE and 85.34% δ1 accuracy, which was trained for 70 epochs, with 0.01 learning rate, L1 (Mean Absolute Error) loss type and 32 batch size. To put this into perspective, RMSE encapsulates the fact that, while making predictions on unseen images, the model predicts a depth, which on average, is half a meter off from the target depth. This is impressive, considering that we only have 1 image to compute the dense depth and how lightweight the network is. Additionally, with increase in batch size, we see a decrease in RMSE score and an increase in δ1 accuracy. This conforms with normal trend because with an increase in batch size, the model gets to see more data before updating parameters, which results in better generalization. However, it is well known that too large of a batch size will lead to poor generalization and the model would overfit the data. This decrease in performance can be observed in the last training trail with batch size 128, where we see an increase in RMSE and decrease in accuracy (something we do not want!). Smaller batch sizes allow the model to learn before seeing all the data. But the downside to this is the model might not converge to the global optima. Therefore, under no computational constraints, it is advised that one starts at a small batch size to reap the benefits of faster training and steadily grow the batch size, also reaping the benefits of faster convergence.

# Results

We present the dense depth maps generated by our model on various indoor scenes.

![res](https://user-images.githubusercontent.com/18104407/146619597-5cc7ab0d-868e-414c-8ed1-00c48b838a31.png)

In the depth maps generated above in figure 8, the objects closest to the vantage point are represented in blue while the objects farthest from the vantage point are represented in yellow. The depth range between these extremes is represented by the colors on the Viridis color palette. It can also be seen that the generated depth maps are accurate. This is also due to additive skip connections from the layers of the encoder network to the layers of the decoder network, which allows image details from high resolution feature maps in the encoder to be merged into features within the decoder. Additionally, the quicker computation of depth-wise convolutions in the encoder architecture results in a faster model, which on average takes 25 ms to generate a depth map on a single 2D image. To put this into context, the model can generate/play real-time depth information up to 33 FPS, to be precise. The following are the Youtube video links:
* Source RGB video (captured from my phone camera): https://www.youtube.com/watch?v=6RbQ1TJZjLE 
* Generated dense depth video: https://www.youtube.com/watch?v=9PHb30Zicos

# References 
* https://paperswithcode.com/task/monocular-depth-estimation/codeless?page=2

* Andrew G. Howard, M. Z. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CORR.

* Ashutosh Saxena, S. H. (2006). Learning Depth from Single Monocular Images. Stanford University.

* Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. 10.1109/CVPR.2017.195, (pp. 1800-1807).

* Dijk, T. v. (2019). How Do Neural Networks See Depth in Single Images?

* Olaf Ronneberger, P. F. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

* DBLP:journals/corr/RonnebergerFB15, abs/1505.04597. Retrieved from http://arxiv.org/abs/1505.04597

* silberman. (n.d.). Retrieved from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

* Wonka, I. A. (n.d.). High Quality Monocular Depth Estimation via Transfer Learning. CoRR. Retrieved from http://arxiv.org/abs/1812.11941
