<div style="text-align: justify;">
# Introduction to Machine Learning

This document provides a foundational understanding of Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), outlining their historical development, core concepts, and interrelationships. It also delves into key aspects of data analysis, including data types, quality, characteristics, and preprocessing techniques, along with methods for calculating dissimilarities

## Machine Learning (ML)

ML addresses the question of whether a computer can "learn on its own how to perform a specified task" by automatically learning data-processing rules from data, rather than being explicitly programmed.

### Essential Components to implement ML

1. Input data points.
2. Examples of the expected output.
3. A method to measure the algorithm's performance.

<u>Examples of ML</u>: Machines playing chess, detecting and deleting spam, language translation.

## ML vs. Classical Statistics Comparison:

| Feature         | Statistical Approach                  | Machine Learning                        | 
| :-------------- | :----------------------------------- | :-------------------------------------- | 
| Approach        | Modeling data generating process      | Algorithmic modeling                    | 
| Driver          | Math, Theory                         | Fitting data using optimization technique| 
| Focus           | Hypothesis testing                   | Predictive accuracy                     | 
| Data Size       | Any reasonable set                   | Large-scale dataset                     | 
| Dimensions      | Mostly for low-dimensional data      | High-dimensional data                   | 
| Inference       | Parameter estimation                 | Prediction                              | 
| Interpretability| High | Medium to low

### Categories of Machine Learning (ML)

Machine learning is primarily divided into three main categories.

#### Supervised Learning
This approach typically involves training a model using input data points and examples of the expected output. The algorithm then learns to map inputs to outputs based on this labeled training data. While not explicitly stated in the sources, the term "supervised" implies the presence of these known outputs or "labels" for the training data.

<b>Key Tasks</b>:

1. <u>Classification</u>: It deals with categorizing data into predefined classes or labels. Examples of applications for Classification include:
    - Image Classification
    - Customer Retention
    - Identity Fraud Detection
    - Diagnostics
2. <u>Regression</u>: Unlike classification, which predicts discrete categories, regression typically involves predicting a continuous value. xamples of applications for Regression include:
    - Advertising Popularity Prediction
    - Weather Forecasting
    - Population Growth Prediction
    - Market Forecasting
    - Estimating life expectancy

#### Unsupervised Learning
Unsupervised learning is about finding hidden patterns in data that has not been labeled. The algorithm explores the data on its own to identify meaningful structures or clusters without any predefined outcomes to guide it.

<b>Key Tasks</b>:

1. <u>Dimensionality Reduction</u>: Dimensionality Reduction is a specific task within Unsupervised Learning that addresses the issue of handling data with a large number of attributes or features. The "dimensions" of a dataset refer to the number of attributes that the data objects possess. For example, the dimensions of a dog's image could include thickness of ear, width of nose, width of dog, diameter of leg, color of fur, and height of dog. The primary objective of dimensionality reduction is to "reduce the data point to a smaller number of informative features". This process can help to "eliminate irrelevant features and reduce noise" within the dataset.
2. <u>Clustering</u>: Clustering is another significant task performed under Unsupervised Learning. Its core purpose is to identify groups or "clusters" of data objects that are similar to each other, based on their inherent characteristics. This analysis can be performed by computing the similarity or distance between pairs of objects.

#### Reinforcement Learning (RL)
Reinforcement learning is about training an agent to make a sequence of decisions. The agent learns through trial and error in an interactive environment. It receives rewards for good actions and penalties for bad ones, with the overall goal of maximizing its total reward over time.

<b>Key Tasks</b>:

1. <u>Real-time Decisions</u>: RL is suited for situations requiring quick and adaptive decision-making in dynamic environments.
2. <u>Game AI</u>: A prominent application of RL involves developing artificial intelligence that can learn to play and master games. This demonstrates RL's capacity for strategic thinking and optimization over time.
3. <u>Robot Navigation</u>: RL is used to enable robots to learn how to move and navigate effectively within their environment.
4. <u>Skill Acquisition</u>: This suggests that RL can be applied to teach machines to acquire various skills through interaction and feedback.
5. <u>Learning Tasks</u>: This is a broader term that encompasses the general learning capabilities enabled by reinforcement learning.

</div>