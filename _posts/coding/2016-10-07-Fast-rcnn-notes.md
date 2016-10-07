---
layout: post
title: fast rcnn org
description: Fast rcnn paper's note
category:  detection
---

# Fast R-CNN --Ross Girshick

Paper: [[http://arxiv.org/abs/1504.08083][Fast R-CNN]]
Code: [[https://github.com/rbgirshick/fast-rcnn][Fast R-CNN's code]]


## Fast R-CNN architecture and training 

   [[./pic_fast_rcnn/1.png]]
   - inputs: #an entire image#, #a set of object proposals#
   - several convolutional and max pooling layers -> produce a conv feature map
   - for each object proposal: a RoI(region of interest) pooling layer extracts a 
     fixed-length feature vector from the feature map
   - each feature vector is fed into a sequence of fully connected(fc) layers 
     that finally branch into two sibling(兄弟，姐妹，同属) output layers:
     #one# that produces softmax probability estimates over K object classes
     plus a catch-all "background" class and #another layer# that outputs 
     four real-valued numbers for each of the K object classes.
   - [ ] ? 2. /Each set of 4 values encodes refined bounding-box positions for one of
           the K classes./

### The RoI pooling layer
    - The RoI pooling layer uses max pooling to convert the features inside any valid
    region of interest into a small feature map with a fixed spatial extent of HxW,
    wherer H and W are layer hyper-parameters that are independent of any particular RoI.

    - RoI: (r,c,h,w) specifies its top-left corner(r,c) and its height and width(h,w).

    - RoI max pooling layer divides the hxw RoI window into an HxW grid of sub-windows of
      approximate size h/H x w/W and then max-pooling the values in each sub-window into 
      the corresponding output grid cell.

### Initializing from pre-trained networks

    - When a pre-trained network initializes a Fast R-CNN network, it undergoes three
      transformations:
      1. The last max pooling layer is replaced by a RoI pooling layer that is configured
         by setting H and W to be compatible with the net's first fully connected layer
         (e.g., H = W = 7 for VGG16).
      2. The network's last fully connected layer and softmax are replaced with the two 
         sibling layers described earlier: a fully connected layer and softmax over K + 1
         categories, category-specific bounding-box regressors.
      3. The network is modified to take two data inputs: a list of images and a list of
         RoIs in those images.

### Fine-tuning for detection

    - In Fast R-CNN training, stochastic gradient descent(SGD) mini-batches are sampled 
      hierarchically.
      1. First sampling N images
      2. Second sampling R/N RoIs from each image.
    - RoIs from the same image share computation and memory in the forward and backward
      passes.
    - One concern over this strategy is it may cause slow training convergence because
      RoIs from the same image are correlated. This concern does not appear to be a 
      practical issue and we achieve good results with N = 2 and R = 128 using fewer
      SGD iterations than R-CNN.

#### Multi-task loss

     - A Fast R-CNN network has two sibling output layers.
       1. The first outputs a discrete probability distribution(per RoI), 
          $p = (p_0, ..., p_K)$, over K + 1 categories.(a softmax over the K + 1 outputs of a
          fully connected layer.
       2. The second sibling layer outputs bounding-box regression offsets, 
          $t^k = (t_x^k, t_y^k, t_w^k, t_h^k)$, for each of the K object classes, indexed by k.
       3. [ ] We use the parameterization for $t^k$ given in [fn:1], in which t^k specifies a
          scale-invariant translation and log-space height/width shift relative to an object 
          proposal.
     - #Each trainging RoI# is labeled with a ground-truth class u and a ground-truth bounding-box
       regression target v. We use a multi-task loss L on each labeled RoI to jointly train for
       classification and bounding-box regression:
       \begin{equation}
         L(p, u, t^u, v) = L_{cls}(p, u) + \lambda[u\ge1]L_{loc}(t^u, v)         
       \end{equation}
       in which $L_{cls}(p, u)  = -logp_u$ is log loss for true class u.
     - The second task loss , $L_loc$, is defined over a tuple of true bounding-box regression 
       targets for class u. The Iverson bracket indicator function $[u\ge1]$ evaluates to 1 when 
       $u>1$ and 0 otherwise.For background RoIs there is no notion of a ground-truth bounding box
       and hence $L_{loc}$ is ignored. For bounding-box regression, we use the loss
       \begin{equation}
         L_{loc}(t^u, v) = \sum_{i\in{x, y, w, h}} smooth_{L_1}(t_i^u - v_i)         
       \end{equation}
       in which 
       \begin{equation}
         smooth_{L_1}(x) = 
       \begin{cases}
       {0.5x^2} &\mbox{if |x| < 1}\\
       {|x| - 0.5} &\mbox{otherwise}
       \end{cases}
       \end{equation}
       is a robust $L_1$ loss that is less sensitive to outliers than the $L_2$ loss used in 
       R-CNN and SPPnet.
       - When the regression targets are unbounded, training with $L_2$ loss can require careful
         tuning of learning rates in order to prevent exploding gradients. Eq.3 eliminates this
         sensitivity.
     - We normalize the ground-truth regression targets $v_i$ to have zero mean and unit variance.
       All experiments use $\lambda = 1$.
     - [fn:2] uses a related loss to train a class agnostic object proposal network. [fn:2] advocates
       for a two-network system that separates localization and classification.
     - OverFeat[fn:3], R-CNN[fn:1], and SPPnet[fn:5] alse train classifiers and bounding-box 
       localizers, however these methods use stage-wise training, which we show is suboptimal
       for Fast R-CNN.

#### Mini-batch sampling

     1. During fine-tuning, each SGD mini-batch is constructed from N = 2 images, chosen uniformly
        at random. We use mini-batches of size R = 128, sampling 64 RoIs from each images.
     2. As in [fn:1], we take 25% of the RoIs from object proposals that have intersection over
        union(IoU) overlap with a ground-truth bounding box of at least 0.5. These RoIs comprise
        the examples labeled with a foreground object class, i.e. $u \ge 1$.
     3. The remaining RoIs are sampled from object proposals that have a maximum IoU with ground truth
        in the interval [0.1, 0.5), following [fn:5].
        1) These are the background examples and are labeled with u = 0.
        2) The lower threshold of 0.1 appears to act as a heuristic for hard example mining [fn:4].
     4. During traing, images are horizontally flipped with probability 0.5. No other data 
        augmentation is used.

#### Back-propagation through RoI pooling layers

     1. The RoI pooling layer's backwards function computes partial derivative of the loss
        function with respect to each input variable $x_i$ by following the argmax switches:
        \begin{equation}
          \frac{\partial{L}}{\partial{x_i}} = \sum_r\sum_j[i = i#(r,j)]\frac{\partial{L}}{\partial{y_{rj}}}
        \end{equation}
        - where $x_i\in{R}$ be the i-th activation input into the RoI pooling layer and 
        $y_{rj}$ be the layer's j-th output from the r-th RoI.
        - The RoI pooling layer computes $y_{rj}=x_{i#(r,j)}$, in which $i#(r,j)=argmax_{i^{'}\in{R(r,j)}}x_{i^{'}}$. 
        $R(r,j)$ is the index set of inputs in the sub-window over which the output unit $y_{rj}$ 
        max pools.

#### SGD hyper-parameters

     - The fully connected layers used for softmax classification and bounding-box regression
       are initialized from $N(0,0.01^2)$ and $N(0,0.001^2)$. Biases are initialized to 0.
     - All layers use a pre-layer learning rate of 1 for weights and 2 for biases and a global
       learning rate of 0.001.
     - When training on VOC07 or VOC12 trainval we run SGD for 30k mini-batch iterations, and
       then lower the learning rate to 0.0001 and train for another 10k iterations.
     - Momentum : 0.9 , Parameter decay : 0.0005(on weights and biases)

### Scale invariance

    1. We explore two ways of achieving scale invariant object detection:
       1) via "brute force"
       2) by using image pyramids
    2. These strategies follow the two approaches in [fn:5].
    3. Brute-force approach
       - Each image is processed at a pre-defined pixel size during both training and testing.
       - The network must directly learn scale-invariant object detection from the training data.
    4. Multi-scale approach
       - Provides approximate scale-invariance to the network through an image pyramid.
       - At test-time, the image pyramid is used to approximately scale-normalize each object 
         proposal.
       - During multi-scale training, we randomly sample a pyramid scale each time an image is
         sampled, following [fn:5], as a form of data augmentation.
    5. We experiment with multi-scale training for smaller networks only, due to GPU memory limits.
          
## Fast R-CNN detection

   - The network takes as input an image(or an image pyramid, encoded as a list of images) and a list
     of R object proposals to score. At test-time, R is typically around 2000, although we will 
     consider cases in which it is larger($\approx45k$).
   - When using an image pyramid, each RoI is assigned to the scale such that the scaled RoI is
     closest to $224^2$ pixels in area [fn:5].
   - For each test RoI r, the forward pass outputs a class posterior probability distribution p and
     a set of predicted bounding-box offsets relative to r(each of the K classes gets its own refined
     bounding-box prediction).
   - We assign a detection confidence to r for each object class k using the estimated probability 
     $P_r(class=k|r)=p_k$.
   - We then perform non-maximum suppression independently for each class using the algorithm and 
     settings from R-CNN[fn:1].

### Truncated SVD for faster detection

   [[./pic_fast_rcnn/2.png]]
   - For whole-image classification, the time spent computing the fully connected layers is small 
     compared to the conv layers. On the contrary, for detection the number of RoIs to process is
     large and nearly half of the forward pass time is spent computing the fully connected layers.
   - Large fully connected layers are easily accelerated by compressing them with truncated 
     SVD[fn:6][fn:7].
   - In this technique, a layer parameterized by the $u\times{v}$ weight matrix W is approximately 
     factorized as
     \begin{equation}
       W\approx{U\sum_tV^T}
     \end{equation}
     In this factorization, U is a $u\times{t}$ matrix comprising the first t left-singular vectors
     of W, $\sum_t$ is a $t\times{t}$ diagonal matrix containing the top t singular values of W,
     and V is $v\times{t}$ matrix comprising the first t right-singular vectors of W.
   - Truncated SVD reduces the parameter count from $uv$ to $t(u+v)$, which can be 
     significant if t is much smaller than min(u,v).
   - To compress a network, the single fully connected layer corresponding to W is replaced
     by two fully connected layers, without a non-linearity between them.
     1) The first of these layers uses the weight matrix $\sum_tV^T$ (and no biases).
     2) The second uses $U$ (with the original biases associated with $W$).
   - This simple compression method gives good speedups when the number of RoIs is large.
     
## Main results

   - Three main results support this paper's contributions:
     1) State-of-the-art mAP on VOC07, 2010, and 2012
     2) Fast training and testing compared to R-CNN, SPPnet
     3) Fine-tuning conv layers in VGG16 improves mAP
### Experimental setup
    - Our experiments use three pre-trained ImageNet models that are available online[fn:8].
      1) The first is the CaffeNet(essentially AlexNet[fn:9]) from R-CNN[fn:1]. We alternatively
         refer to this CaffeNet as model $S$, for "small".
      2) The second network is VGG_CNN_M_1024 from [fn:10], which has the same depth as $S$,
         but is wider. We call this network model $M$, for "medium".
      3) The final network is the very deep VGG16 model from [fn:11]. We call  it model $L$.
    - In this section, all experiments use single-scale training and testing(s=600).

### VOC 2010 and 2012 results
    
### VOC 2007 results

### Training and testing time

    - Fast training  and testing times are our second main result.

      [[./pic_fast_rcnn/table4.png]]

#### Truncated SVD

     - Truncated SVD can reduce detection time by more than 30% with only a small drop 
       in mAP and without needing to perform additional fine-tuning after model compression.
     - Using the top 1024 singular values from the $25088\times{4096}$ matrix in VGG16's fc6 layer
       and the top 256 singular values from the $4096\times{4096}$ fc7 layer reduces runtime
       with little loss in mAP.

       [[./pic_fast_rcnn/2.png]]


### Which layers to fine-tune?

    - Our hypothesis: training through the RoI pooling layer is important for very deep nets.

      [[./pic_fast_rcnn/table5.png]]

    - Does this mean that all conv layers should be fine-tuned?
      In short, no.
      1) In the smaller networks $S$ and $M$ , we find that conv1 is generic and task 
         independent(a well-known fact)[fn:12]. Allowing conv1 to learn, or not, has no
         meaningful effect on mAP.
      2) For VGG16, we found it only necessary to update layers from conv3_1 and up(9 of the 13
         conv layers).
      3) This observation is pragmatic:
         1. updating from conv2_1 slows trainging by 1.3x (12.5 vs. 9.5 hours) compared to 
            learning from conv3_1
         2. Updating from conv1_1 over-runs GPU memory
      4) All Fast R-CNN results in this paper using VGG16 fine-tune layers conv3_1 and up;
         all experiments with models $S$ and $M$ fine-tune layers conv3 and up.
         
## Design evaluation

   - We conducted experiments to understand how Fast R-CNN compares to R-CNN and SPPnet, as well 
     as to evaluate design decisions.
   
### Does multi-task training help?

    - We observe that multi-task training improves pure classification accuracy relative to
      training for classification alone.
    - Stage-wise training improves mAP over column one, but underperforms multi-task training.
    
      [[./pic_fast_rcnn/table6.png]]

### Scale invariance : to brute force or finesse?

    - We compare two strategies for achieving scale-invariant object detection:
      brute-force learning(single scale) and image pyramids(multi-scale). In either
      case, we define the scale s of an image to be the length of its shortest side.
    - All single-scale experiments use s = 600 pixels.
    - In the multi-scale setting, we use the same five scales specified in [fn:5]
      $s\in{{480,576,688,864,1200}}$ to facilitate comparison with SPPnet.
      [[./pic_fast_rcnn/table7.png]]
    - Deep ConvNets are adept at directly learning scale invariance.
    - The multi-scale approach offers only a small increase in mAP at a large cost
      in compute time.

### Do we need more training data?

### Do SVMs outperform softmax?

    - Fast R-CNN uses the softmax classifier learnt during fine-tuning instead of
      training one-vs-rest linear SVMs post-hoc, as was done in R-CNN and SPPnet.
      [[./pic_fast_rcnn/table8.png]]
    - Softmax slightly outperforming SVM for all three networks.
    - This effect is small, but it demonstrates that "one-shot" fine-tuning is sufficient
      compared to previous multi-stage training approaches.
    - We note that softmax, unlike one-vs-rest SVMs, introduces competition between classes
      when scoring a RoI.

### Are more proposals always better?

    - There are two types of object detectors : those that use a sparse set of object 
      proposals[fn:13] and those that use a dense set DPM[fn:14].

### Preliminary MS COCO results

## Conclusion 

   - This paper proposes Fast R-CNN, a clean and fast update to R-CNN and SPPnet.
   - Of particular note, sparse object proposals appear to improve detector quality.
   - There may exist yet undiscovered techniques that allow dense boxes to perform 
     as well as sparse proposals.

