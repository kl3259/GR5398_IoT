# **GR5398-IoT**


The project is aimed at developing new tools for classifying videos of human-machine interactions in the Internet-of-Things (IOT) domain. Namely, given videos of humans interacting with IoT devices (i.e. smart appliances such as fridge, toaster, washing machines, Alexa, etc), the aim is to (1) design predictive video features, which (2) can be extracted efficiently in real-time to classify videos in terms of the activity being performed (opening or closing a fridge, loading or unloading a washing machine, etc.). The grand goal and motivation for the work is to generate labels for IoT network traffic, simply by training cameras onto IoT devices in the various IoT labs across US universities. Thus, the project aims to solve a main bottleneck in research at the intersection of Machine Learning and IoT, namely, the scarcity of labeled IoT traffic data to solve ML problems such as activity and anomaly detection using supervised or unsupervised detection procedures. 


### **Week 05/24/2022**
* Make sanity checks using synthetic data w/ 2-3 core features and 5-10 redundant features
  * Syncthetic data
  * Logistic Regression
  * Model Comparison: LR + all features b/w LR + directions selected by SAVE
* Figure out why SAVE returns the ratio of top1 eigenvalue to top2 eigenvalue close to 1 (should be large)
* Plot learning curves w/ n_directions as x-axis
* Brief report


### **Week 05/31/2022**
* Fix issue in SAVE function
* All-In-One learning curve
  *  GridSearch best parameters when plotting learning curve
* Check basis by comparing distances / angles
* Transformed data length


### **Week 07/05/2022**
* SAVE can find the sufficient direction, but there is no improvement for the classification
  * Change classifier
* Consider dimension reduction for multivariate longitudinal data
  * Latent Markov Model
    * Define optimal one-dimensinal summaries and the orthonomal weight space


### **Week 07/11/2022**
* **Multivariate Time Series Classification (MTSC)**
  * _discriminatory features may be in the interactions between dimensions, not just in the autocorrelation within an individual series_
  * Approaches:
    * Deep learning based methods
    * Shaplets method
    * Bag of Words approaches
    * Distance based approaches: Dynamic Time Warping -> benchmark
    * Ensemble univariate TSC classifiers over the multivariate dimensions

### **Dynamic Time Warping**
* Intuition: 1NN classifier + specific distance function + series re-alignment
* Can be used in unequal series
* **MTSC**: 
  
  $x_{i,j,k}$ denote the $i$ th case (data points), $j$ th observation (time steps) of dimension $k$ (different predictors)

  Let $m$ denote the number of time steps and $d$ denote the number of predictors

  Assume that the correct warping is the same across all dimensions -> distance is like Euclidean distance between 2 vectors of all dimensions from MTSC $x_a$ and $x_b$ 

**_Algorithm: Dynamic Time Warping_**
1. Calculate distance matrix $M$
   $$M_{i,j}(\bf{x_a}, \bf{x_b}) = \sum_{k = 1}^{d} (x_{a,i,k} - x_{b,j,k})^2$$
2. Define warping path $P = ((e_1, f_1), ... , (e_s, f_s))$ subject to constratints:
   * $(e_1, f_1) = (1,1)$ and $(e_s, f_s) = (m,m)$
   * $0 \leq e_{i+1} - e_i \leq 1$ for all $1 \leq i \leq m$
   * $0 \leq f_{i+1} - f_i \leq 1$ for all $1 \leq i \leq m$
3. Let $p_i = M_{e_i, f_i}$ as the step $i$ in path, the total distance for a path is $D_p = \sum_{i=1}^{m}p_i$ 
4. Find the path w/ minimal accumulative distance $P^* = min_{p \in P} D_P(x_a,x_b)$ 
5. Get the optimal distance by **Dynamic Programming**: 
   
   ![](https://latex.codecogs.com/svg.image?%5Clarge%20DTW(i,j)%20=%20M_%7Bi,j%7D%20&plus;%20min%5Cleft%5C%7B%5Cbegin%7Baligned%7D&%20DTW(i-1,%20j),%20%5C%5C&%20DTW(i,%20j-1),%20%5C%5C&%20DTW(i-1,%20j-1).%20%5C%5C%5Cend%7Baligned%7D%5Cright.)

   return final distance: $DTW(m,m)$ 


### **Week 07/18/2022**

### **Random Convolutional Kernel Transform (ROCKET)**
* Feature capture by convolutional kernels
  * Effective in CNN like ResNet and InceptionTime
* Intuition: Kernels can detect patterns in time series despite warping. Pooling mechanisms make kernels invariant to the position of patterns in time series. Dilation allows kernels with similar weights to capture patterns at different scales, i.e., despite rescaling. Multiple kernels with different dilations can, in combination, capture discriminative
patterns despite complex warping. 
* Use a large number of random convolutional kernels, default number is 10,000
* Randomization: random length, weights, bias, dilation, and padding
* Classifier: Ridge Regression / Logistic Regression
  * Receive transformed features as input
* Similar feature maps like global max pooling
* Introduced **proportion of positive values (ppv)**, which enables a classifier to weight the prevalence of a given pattern within a time series
* When using Logistic Regression as classifier, ROCKET actually becomes a single layer CNN w/o nonlinearities

**_Algorithm: ROCKET_**
1. Determine number of kernels $N$, and number of features per kernel $n$
2. Generate kernels, for each kernel:
   * Length: $l = \{7,9,11\}$ w/ equal probability
   * Weights: $w \sim N(0,1)$, then pick the mean centered weight $w = w - \bar{W}$
   * Bias: $b \sim U(-1, 1)$
   * Dilation: $d = \lfloor 2^x \rfloor, x \sim U(0, A), A = \log_2{\frac{l_{input} - 1}{l_{kernel} - 1}}$
   * Padding: randomly determine whether using padding or not(50% / 50%) for each kernel, if used, $p= (l_{kernel} - 1) \times d / 2$
   * Stride: $s = 1$
3. Transformation
   * For kernel $w \in W$ and time step $i$ in time series $X$, for each kernel $w$ w/ dilation $d$ bias $b$ applying to a time series $X$, the convolution operation is: 
     $$X_i \cdot w = b + (\sum_{j = 0}^{l_{kernel} - 1} X_{i + (j \times d)} \times w_j)$$
   * Compute aggregate features, 10,000 kernels -> 20,000 features:
     1. maximum value (maxpooling)
     2. proportion of positive values (ppv): Assume that $Z$ is the output of convolution operation, 
        $$ppv(Z) = \frac{1}{n}\sum_{i = 0}^{n-1} I(z_i > 0)$$
4. Classification
   * Ridge Regression for n_samples < n_features, other wise use Logistic Regression
   * Cross validation for hyperparameters

### **Week 07/25/2022**
* Information Exchange Block (IEBlock)








