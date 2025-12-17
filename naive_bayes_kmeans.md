# Naive Bayes & K-Means Lecture Notes

## Lecture 1: Naive Bayes

The Starting Point of Probabilistic Modeling

### 1. Start from the Classification Problem

Given an input x, the output is a class label y:

$$
y \in {1,2,\dots,K}
$$

The goal is:

$$
\hat{y} = \arg\max_y P(y|x)
$$

This is the common objective of all classification models.

### 2. Bayes’ Rule (Core Formula)

By Bayes’ theorem:

$$
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

Since P(x) does not depend on y:

$$
\hat{y} = \arg\max_y P(x|y)P(y)
$$
 • Discriminative models: directly model P(y|x)
 • Generative models: model P(x|y) and P(y)

Naive Bayes is a generative model.

### 3. Conditional Independence Assumption (Key Assumption)

Let:

$$
x = (x_1, x_2, \dots, x_d)
$$

Naive Bayes assumes conditional independence:

$$
P(x|y) = \prod_{i=1}^{d} P(x_i|y)
$$

Three key points to emphasize:

 1. This assumption is almost certainly wrong.
 2. But it dramatically reduces model complexity.
 3. It makes learning computationally feasible.

This illustrates an important idea in machine learning:

Modeling assumptions ≠ the real world.

### 4. Parameter Learning (Discrete Features)

For each class y, we estimate:

$$
P(x_i = v | y)
$$

Using Maximum Likelihood Estimation (MLE):

$$
P(x_i=v|y) = \frac{\text{count}(x_i=v, y)}{\text{count}(y)}
$$

Laplace smoothing is often used:

$$
P(x_i=v|y) = \frac{n_{iv}+1}{n_y + |V|}
$$

### 5. Common Variants of Naive Bayes

1. Bernoulli Naive Bayes
 • Features are binary (0/1)
 • Whether a feature appears

2. Multinomial Naive Bayes
 • Uses word counts / frequencies
 • Most common for text classification

3. Gaussian Naive Bayes
 • Continuous features
 • Assumes:

$$
x_i|y \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)
$$

### 6. Decision Function (Log-Space)

To avoid numerical underflow, we use log-space:

$$
\log P(y|x) \propto \log P(y) + \sum_i \log P(x_i|y)
$$

### 7. Advantages and Disadvantages (Class Discussion)

Advantages
 • Extremely fast
 • Very few parameters
 • Robust for small datasets
 • Clear theoretical foundation

Disadvantages
 • Independence assumption is too strong
 • Limited expressive power
 • Cannot model complex feature interactions

### 8. Summary

Naive Bayes = using a very strong independence assumption to turn an otherwise intractable probabilistic problem into one that can be solved by simple counting.

**I. What Problem Does Naive Bayes Solve?**
We want to classify:
 • Input: x = (x_1, x_2, \dots, x_d)
 • Output: class label y

Goal:

$$
\hat{y} = \arg\max_y P(y|x)
$$

**II. Why Is Directly Modeling P(y|x) Hard?**
Curse of dimensionality: estimating a high-dimensional distribution requires far more data than we usually have.

**III. Bayes’ Rule as a Change of Perspective**
We rewrite:

$$
\hat{y} = \arg\max_y P(x|y)P(y)
$$

We shift from:
 • “Given x, which class y is it?”
to
 • “If y were known, what kind of x would it generate?”

This is the core idea of a generative model.

**IV. The Real Difficulty: P(x|y) Is Still Intractable**
$$
P(x|y) = P(x_1, x_2, \dots, x_d | y)
$$

Even for modest d, the number of parameters explodes.

**V. The “Fatal” Assumption (But Also the Key)**
$$
P(x|y) = \prod_{i=1}^{d} P(x_i|y)
$$

Emphasize:

 1. almost surely false
 2. makes the model feasible
 3. works surprisingly well

**VI. Why Does It Still Work (Intuition)?**
In text classification, words are correlated in reality, but NB assumes independence.
It can still work because classification often only needs relative scores, not perfectly calibrated probabilities, and errors may cancel across classes.

**VII. How Are Parameters Learned?**
Prior probability (counting):

$$
P(y=k) = \frac{\text{count}(y=k)}{N}
$$

Conditional probability (still counting), Bernoulli NB:

$$
P(x_i=1|y) = \frac{\text{count}(x_i=1,y)+1}{\text{count}(y)+2}
$$

(Laplace smoothing prevents zero probabilities.)
Learning is counting + normalization; no gradient descent.

**VIII. Why Use Log at Prediction Time?**
$$
P(y|x) \propto P(y)\prod_i P(x_i|y)
$$

Products underflow, so we use logs:

$$
\log P(y|x) = \log P(y) + \sum_i \log P(x_i|y)
$$

**IX. What Is Naive Bayes Really Doing?**

 • Each feature provides a “vote” (log-likelihood contribution)
 • Sum votes across features
 • Choose the class with the highest score

This is structurally similar to Logistic Regression, MLPs, and linear classifiers.

## Lecture 2: k-Means Clustering

The Starting Point of Unsupervised Learning and Representation Learning

### 1. Clustering Problem Definition

Given data:

$$
{x_1, x_2, \dots, x_N}, \quad x_i \in \mathbb{R}^d
$$

Goal: partition the dataset into K clusters.

### 2. k-Means Objective Function

$$
\min_{{c_i}, {\mu_k}} \sum_{i=1}^{N} |x_i - \mu_{c_i}|^2
$$

This is a geometric objective, not a probabilistic model.

Where:
 • $x_i$: the i-th sample
 • $c_i \in \{1,\dots,K\}$: cluster assignment
 • $\mu_k$: centroid of cluster k

k-means is not “guessing clusters”; it is optimizing a clear objective.

### 3. Lloyd’s Algorithm (Core Algorithm)

- **Step 1: Assignment**

$$
c_i = \arg\min_k |x_i - \mu_k|^2
$$

- **Step 2: Update**

$$
\mu_k = \frac{1}{|\mathcal{C}k|} \sum{i:c_i=k} x_i
$$
 • Each step decreases the objective
 • But the overall problem is non-convex

### 4. Why Does k-Means Converge?

Because:
 • The assignment step does not increase the objective
 • The update step is optimal given current assignments (the mean)

However:
 • It only guarantees convergence to a local optimum.

### 5. Why Squared Euclidean Distance?

Two key reasons:

1) Simple geometric meaning
 • Euclidean distance = straight-line distance
 • Squaring penalizes far-away points more

2) Mathematically convenient
For a fixed cluster C_k, minimizing

$$
\sum_{x \in C_k} |x - \mu|^2
$$

gives the closed-form solution

$$
\mu = \frac{1}{|C_k|}\sum_{x \in C_k} x
$$

That is why the centroid is the mean—this is derived, not arbitrary.

### 6. Initialization Sensitivity

k-means is extremely sensitive to initialization because:
 • the objective is non-convex
 • there are many local minima

Solutions:
 • multiple random restarts
 • k-means++

k-means++ intuition:
 • first centroid random
 • later centroids sampled proportional to distance from existing centroids
Goal: spread initial centroids apart.

### 7.  Advantages and Disadvantages

Advantages
 • simple and intuitive
 • fast
 • interpretable
 • excellent for teaching

Disadvantages
 • must choose K in advance
 • sensitive to initialization
 • sensitive to outliers(异常点)

### 8. KMeans Summary

k-means teaches us:

If you clearly specify what kind of clusters you want, the algorithm can be very simple.
But if real data violates those assumptions, simple algorithms will fail.
