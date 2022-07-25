# **GR5398-IoT**


The project is aimed at developing new tools for classifying videos of human-machine interactions in the Internet-of-Things (IOT) domain. Namely, given videos of humans interacting with IoT devices (i.e. smart appliances such as fridge, toaster, washing machines, Alexa, etc), the aim is to (1) design predictive video features, which (2) can be extracted efficiently in real-time to classify videos in terms of the activity being performed (opening or closing a fridge, loading or unloading a washing machine, etc.). The grand goal and motivation for the work is to generate labels for IoT network traffic, simply by training cameras onto IoT devices in the various IoT labs across US universities. Thus, the project aims to solve a main bottleneck in research at the intersection of Machine Learning and IoT, namely, the scarcity of labeled IoT traffic data to solve ML problems such as activity and anomaly detection using supervised or unsupervised detection procedures. 

<br>


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
* Transformed data length?


### **Week 07/05/2022**
* Consider dimension reduction for multivariate longitudinal data
  * Latent Markov Model
    * Define optimal one-dimensinal summaries and the orthonomal weight space


### **Week 07/10/2022**
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
   
   ![](https://latex.codecogs.com/gif.image?\large&space;\dpi{120}DTW(i,j)&space;=&space;M_{i,j}&space;&plus;&space;min\left\{\begin{aligned}&&space;DTW(i-1,&space;j),&space;\\&&space;DTW(i,&space;j-1),&space;\\&&space;DTW(i-1,&space;j-1).&space;\\\end{aligned}\right.)

   return final distance: $DTW(m,m)$ 

