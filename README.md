# **GR5398-IoT**

<br>

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

#### **Dynamic Time Warping**
* Intuition: 1NN classifier + specific distance function + series re-alignment
* Can be used in unequal series


