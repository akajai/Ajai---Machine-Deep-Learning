<div style="text-align: justify;">

## Introduction to Machine Learning & Data Analyses

This document provides a foundational understanding of Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), outlining their historical development, core concepts, and interrelationships. It also delves into key aspects of data analysis, including data types, quality, characteristics, and preprocessing techniques, along with methods for calculating dissimilarities

### Machine Learning (ML)

ML addresses the question of whether a computer can "learn on its own how to perform a specified task" by automatically learning data-processing rules from data, rather than being explicitly programmed.

#### Essential Components to implement ML

1. Input data points.
2. Examples of the expected output.
3. A method to measure the algorithm's performance.

**Examples of ML**: Machines playing chess, detecting and deleting spam, language translation.

### ML vs. Classical Statistics Comparison:

| Feature         | Statistical Approach                  | Machine Learning                        | 
| :-------------- | :----------------------------------- | :-------------------------------------- | 
| Approach        | Modeling data generating process      | Algorithmic modeling                    | 
| Driver          | Math, Theory                         | Fitting data using optimization technique| 
| Focus           | Hypothesis testing                   | Predictive accuracy                     | 
| Data Size       | Any reasonable set                   | Large-scale dataset                     | 
| Dimensions      | Mostly for low-dimensional data      | High-dimensional data                   | 
| Inference       | Parameter estimation                 | Prediction                              | 
| Interpretability| High | Medium to low

#### Categories of Machine Learning (ML)

Machine learning is primarily divided into three main categories.

#### Supervised Learning
This approach typically involves training a model using input data points and examples of the expected output. The algorithm then learns to map inputs to outputs based on this labeled training data. While not explicitly stated in the sources, the term "supervised" implies the presence of these known outputs or "labels" for the training data.

<b>Key Tasks</b>:

1. **Classification**: It deals with categorizing data into predefined classes or labels. Examples of applications for Classification include:
    - Image Classification
    - Customer Retention
    - Identity Fraud Detection
    - Diagnostics
2. **Regression**: Unlike classification, which predicts discrete categories, regression typically involves predicting a continuous value. xamples of applications for Regression include:
    - Advertising Popularity Prediction
    - Weather Forecasting
    - Population Growth Prediction
    - Market Forecasting
    - Estimating life expectancy

#### Unsupervised Learning
Unsupervised learning is about finding hidden patterns in data that has not been labeled. The algorithm explores the data on its own to identify meaningful structures or clusters without any predefined outcomes to guide it.

<b>Key Tasks</b>:

1. **Dimensionality Reduction**: Dimensionality Reduction is a specific task within Unsupervised Learning that addresses the issue of handling data with a large number of attributes or features. The "dimensions" of a dataset refer to the number of attributes that the data objects possess. For example, the dimensions of a dog's image could include thickness of ear, width of nose, width of dog, diameter of leg, color of fur, and height of dog. The primary objective of dimensionality reduction is to "reduce the data point to a smaller number of informative features". This process can help to "eliminate irrelevant features and reduce noise" within the dataset.
2. **Clustering**: Clustering is another significant task performed under Unsupervised Learning. Its core purpose is to identify groups or "clusters" of data objects that are similar to each other, based on their inherent characteristics. This analysis can be performed by computing the similarity or distance between pairs of objects.

#### Reinforcement Learning (RL)
Reinforcement learning is about training an agent to make a sequence of decisions. The agent learns through trial and error in an interactive environment. It receives rewards for good actions and penalties for bad ones, with the overall goal of maximizing its total reward over time.

<b>Key Tasks</b>:

1. **Real-time Decisions**: RL is suited for situations requiring quick and adaptive decision-making in dynamic environments.
2. **Game AI**: A prominent application of RL involves developing artificial intelligence that can learn to play and master games. This demonstrates RL's capacity for strategic thinking and optimization over time.
3. **Robot Navigation**: RL is used to enable robots to learn how to move and navigate effectively within their environment.
4. **Skill Acquisition**: This suggests that RL can be applied to teach machines to acquire various skills through interaction and feedback.
5. **Learning Tasks**: This is a broader term that encompasses the general learning capabilities enabled by reinforcement learning.


### Data Analyses
Data analysis involves understanding the nature and quality of data, and preparing it for effective use in ML and DL models.

**Types of Data**

1. Quantitative (Numerical) Data: "measured in numbers."
    - Discrete: Non-decimal numbers (e.g., number of students, number of steps).
    - Continuous: Decimal numbers (e.g., height, width, speed, area).
2. Qualitative (Categorical) Data: "cannot be measured in numbers, but is characterized by similarity among data points."
    - Ordinal: Values can be text or numbers with "internal ordering" (e.g., first/second/third, grades A/B/C).
    - Nominal: Values can be text or numbers with "no ordering" (e.g., dog/cat/elephant, black/blue/red, Asian/African/American).


**Data Quality**

Understanding and improving data quality typically improves the quality of the resulting analysis.

Key Quality Issues: Noise, outliers, missing data, inconsistent data, duplicate data, and biased or unrepresentative data.

- Data Quality Factors:Completeness: Is all necessary data present?
- Timeliness: Is the data available when needed?
- Integrity: Are relations between entities and attributes consistent?
- Validity: Are data values within specified domains?
- Accuracy: Does data reflect real-world objects or a verifiable source?
- Consistency: Is data consistent between systems?
- Uniqueness: Do duplicate records exist?

**General Characteristics of Data Sets**

- **Dimensionality**: The number of attributes an object possesses. High dimensionality can lead to "the curse of dimensionality," necessitating "dimensionality reduction" in preprocessing.
- **Sparsity**: When most attributes of an object have 0 values (e.g., fewer than 1% non-zero). This is an advantage for storage and computation time.
- **Resolution**: The level at which data is obtained. Data properties and patterns vary at different resolutions (e.g., monthly rainfall vs. daily temperature, image resolution).

Terms describe the quality of the measurement process and resulting data.

- **Precision**: "The closeness of repeated measurements (of the same quantity) to one another." Often measured by standard deviation.
- **Bias**: "A systematic variation of measurements from the quantity being measured." Measured by the difference between the mean of values and the true value.
- **Accuracy**: A general term referring to "the degree of measurement error in data," encompassing both precision and bias.


**Data Preprocessing**

- **Goal**: To "improve the data mining analysis with respect to time, cost, and quality."
- **Techniques**:Aggregation: Combining two or more objects into a single object.
- **Sampling**: Selecting a subset of data objects for analysis to estimate characteristics of the whole population (e.g., Simple Random Data Sampling).
- **Dimensionality Reduction**: Eliminating irrelevant features and reducing noise by reducing the number of attributes.
- **Feature Subset Selection**: Reducing dimensionality by selecting a subset of features, especially when redundant or irrelevant features are present.
- **Feature Creation**: Generating new, more effective attributes from original ones, which is highly domain-specific (e.g., extracting features from images like number of white pixels or corner points).
- **Outlier Removal**: Identifying and removing objects that "deviate significantly from the rest" as they can negatively "effect the feature learning capability of the ML algorithms."

Example Preprocessing Pipeline (Iris image recognition):
- Read RGB images.
- Resize images.
- Convert to grayscale images.
- Normalize pixel values to (0,1).
- Centering around 0 (mean=0).
- Standardization (making unit variance).
- Feature creation through vectorization.
- Outlier removal.
- Train the classifier.

**Dissimilarities Between Data Objects (Distance Metrics)**

Methods to find relationships among data objects based on their similarity or distance.

- **Manhattan Distance ($L_1$-norm)**: Manhattan distance, also known as the L1-norm or taxicab distance, is a way of measuring the distance between two points by summing the absolute differences of their coordinates.

    The name "Manhattan distance" comes from the grid-like layout of streets in Manhattan, New York. Imagine you're in a city like Delhi and want to get from point A to point B. You can't fly in a straight diagonal line because buildings are in the way. Instead, you have to travel along the horizontal and vertical streets.

    The total distance you walk along this grid is the Manhattan distance. It's simply the sum of the horizontal distance and the vertical distance.

    For two points in a 2D space, P1 = (x1, y1) and P2 = (x2, y2), the formula is:

    Distance = |x₁ – x₂| + |y₁ – y₂|

    The vertical bars | | represent the absolute value, meaning you only consider the positive difference (distance is always positive).

    Simple Example:
    
    Let's find the Manhattan distance between Point A at (2, 3) and Point B at (5, 7).

    Horizontal distance: |2 – 5| = |-3| = 3 blocks

    Vertical distance: |3 – 7| = |-4| = 4 blocks

    Total Manhattan Distance: 3 + 4 = 7

- **Euclidean Distance ($L_2$-norm)**: The standard straight-line distance between two points in n-dimensional space.

    Think of finding the distance between two landmarks in Delhi, like India Gate and Humayun's Tomb. The Euclidean distance is the length of the perfectly straight line you could draw connecting them on a map, ignoring all the roads, buildings, and other obstacles. It's often called the "as the crow flies" distance
    
    Formula: $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{k=1}^{n} (x_k - y_k)^2}$

    Let's find the Euclidean distance between Point A at (2, 3) and Point B at (5, 7).

    Find the horizontal difference: (5 – 2) = 3

    Find the vertical difference: (7 – 3) = 4

    Square them, add them, and take the square root:
    Distance = √(3² + 4²) = √(9 + 16) = √25 = 5

    **Properties**

    - **Positivity**: $d(\mathbf{x}, \mathbf{y}) \ge 0$, and $d(\mathbf{x}, \mathbf{y}) = 0$ only if $\mathbf{x} = \mathbf{y}$.
    - **Triangular Inequality**: $d(\mathbf{x}, \mathbf{z}) \le d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$.
    - **Symmetry**: $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$.

- **Chebyshev Distance ($L_{\infty}$-norm)**: Minkowski distance with P=$\infty$. "The maximum difference between any attribute of the objects." It's "the maximum distance along one axis."

    The best way to understand Chebyshev distance is to think about a king on a chessboard. A king can move one square in any direction—horizontally, vertically, or diagonally. The Chebyshev distance between two squares is the minimum number of moves a king would need to travel between them.

    For example, if a king needs to move 4 squares horizontally and 2 squares vertically, it can make 2 diagonal moves and 2 horizontal moves, for a total of 4 moves. This is simply the maximum of the two distances (max(4, 2) = 4)

    For two points in a 2D space, P1 = (x1, y1) and P2 = (x2, y2), the formula is:

    Distance = max( |x₁ – x₂|, |y₁ – y₂| )

    This means you calculate the absolute difference for the x-coordinates and the absolute difference for the y-coordinates, and then you simply take the larger of those two values.

    Simple Example:
    Let's find the Chebyshev distance between Point A at (2, 3) and Point B at (5, 7).

    Horizontal distance: |2 – 5| = 3

    Vertical distance: |3 – 7| = 4

    Chebyshev Distance: max(3, 4) = 4

- **Minkowski Distance**: Minkowski distance is a generalized metric that represents the distance between two points in a multi-dimensional space. It's not a unique type of distance itself, but rather a flexible "parent" formula that can become other common distance metrics

    Formula: $d(\mathbf{x}, \mathbf{y}) = \left(\sum_{k=1}^{n} |x_k - y_k|^P\right)^{1/P}$



### Next Topic --> [Pattern Recognition](./PatternReocognition.md)
</div>