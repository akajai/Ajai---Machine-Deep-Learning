<div style="text-align: justify;">

## PCA - Principal Component Analyses

Principal Component Analysis (PCA) is a technique used to simplify complex data by reducing its number of dimensions or variables. In essence, it finds the most important patterns in the data and uses them to create new, summary variables called principal components.

Think of it as creating a high-quality summary of a very long book. You lose some of the minor details, but you keep the main plot and characters, making it much easier to understand.

**An Intuitive Analogy: Describing a Cup of Coffee ☕**

Imagine you want to describe a cup of coffee using several features: temperature, caffeine level, bitterness, sweetness, roast level, and acidity. You have six dimensions (variables) to describe your coffee.

You might notice that some of these variables are related. For example:

- A darker roast level is often associated with higher bitterness.
- Higher temperature might be related to lower perceived sweetness.

Instead of using all six variables, what if you could create a new, more efficient set of variables? PCA does exactly this.

It might find a new variable, let's call it "Strength," that captures the combined effect of roast level and bitterness. This new variable, "Strength," becomes your first principal component. It's the direction in your data that shows the most variation—it does the best job of telling different coffees apart.

Next, PCA looks for a second direction that captures the next most significant variation, making sure it's completely unrelated (orthogonal) to the first one. This might be a new variable called "Flavor Profile," which could capture the balance between sweetness and acidity. This is your second principal component.

By using just these two new variables ("Strength" and "Flavor Profile"), you can describe most of the important differences between the coffees without needing to track all six original features. You have successfully reduced the complexity from six dimensions to two.

#### How PCA Works: A Step-by-Step Overview

PCA finds these new axes (principal components) by looking at the "spread" or variance of the data.

1. Find the Direction of Most Variance: PCA first identifies the direction in the data where the points are most spread out. This direction becomes the First Principal Component (PC1). It's the most important new axis because it captures the largest amount of information (variance).
2. Find the Next Best Direction: Next, it finds a second direction that is perpendicular (orthogonal) to the first and captures the next highest amount of variance. This becomes the Second Principal Component (PC2).
3. Repeat: This process continues for as many dimensions as there are original variables, with each new principal component being orthogonal to all previous ones and capturing the next largest chunk of the remaining variance.

The result is a new set of coordinates for your data based on these principal components. The key is that the first few principal components will contain most of the important information, allowing you to safely ignore the later ones.

### Why Use PCA?

- Dimensionality Reduction: This is the primary goal. Reducing the number of variables simplifies your model, can make it faster to train, and can help avoid the "curse of dimensionality" where models perform poorly in high-dimensional space. In our coffee example, we went from 6 variables to 2.
- Data Visualization: It's impossible for humans to visualize data in more than three dimensions. PCA allows you to reduce high-dimensional data (e.g., 50 variables) down to two or three principal components, which you can then plot and visually inspect for patterns, clusters, or outliers.
3. Noise Reduction: By focusing on the components with the most variance, PCA can help filter out the "noise" in your data, which is often captured by the later, less significant components.

In summary, PCA is a powerful tool for simplifying complex datasets by creating a new set of summary variables that capture the most important patterns, making the data easier to work with and visualize.

To compute the correlation between two features, you're essentially measuring how much they move together in a predictable way. The most common method for this is the Pearson correlation coefficient, which gives you a value between -1 and +1.

Here’s what the values mean:

- +1: A perfect positive correlation. When one feature increases, the other feature increases by a proportional amount.
- 0: No correlation. The two features have no predictable relationship.
- -1: A perfect negative correlation. When one feature increases, the other feature decreases by a proportional amount.

**The Core Idea: Covariance and Standardization**

In layman's terms, the calculation involves two main ideas:

1. Covariance: First, you figure out if the two features tend to move in the same direction (positive covariance) or in opposite directions (negative covariance). To do this, you look at each data point and see if both features are above their respective averages or below their averages. If they're usually on the same side of the average together, the covariance is positive. If they're on opposite sides, it's negative.
2. Standardization: Covariance is great, but its value can be anything (e.g., 5, 500, or -2000), which makes it hard to interpret. To fix this, you standardize the covariance by dividing it by the standard deviation of each feature. The standard deviation is just a measure of how spread out the data for a single feature is. This step scales the result down to the easy-to-interpret range of -1 to +1.

So, Correlation = (How much two features move together) / (How much each feature moves on its own).

**A Simple Example: Ice Cream Sales and Temperature**

Let's say you've collected data for five days on daily temperature and the number of ice cream cones you sold.

| Day | Temperature (°C) (Feature X) | Ice Cream Sales (Feature Y) |
|---|---|---|
| 1 | 20 | 30 |
| 2 | 25 | 40 |
| 3 | 30 | 55 |
| 4 | 35 | 60 |
| 5 | 40 | 65 |

Here’s how you would compute the correlation step-by-step:

- Step 1: Find the Average of Each Feature
    - Average Temperature (xˉ): (20 + 25 + 30 + 35 + 40) / 5 = 30°C
    - Average Sales (yˉ​): (30 + 40 + 55 + 60 + 65) / 5 = 50 cones
- Step 2: Calculate the Deviations for Each Data Point
    
    For each day, find out how far the temperature and sales are from their respective averages.

    | Day | Temp Deviation (x - x̄) | Sales Deviation (y - ȳ) |
    |---|---|---|
    | 1 | 20 - 30 = -10 | 30 - 50 = -20 |
    | 2 | 25 - 30 = -5 | 40 - 50 = -10 |
    | 3 | 30 - 30 = 0 | 55 - 50 = 5 |
    | 4 | 35 - 30 = 5 | 60 - 50 = 10 |
    | 5 | 40 - 30 = 10 | 65 - 50 = 15 |

    Notice that when the temperature is below average (negative deviation), sales are also below average (negative deviation), and vice-versa. This is a hint that we have a positive correlation.
- Step 3: Calculate the Covariance

    Multiply the deviations for each day and find their average.

    | Day | Product of Deviations |
    |---|---|
    | 1 | (-10) * (-20) = 200 |
    | 2 | (-5) * (-10) = 50 |
    | 3 | 0 * 5 = 0 |
    | 4 | 5 * 10 = 50 |
    | 5 | 10 * 15 = 150 |
    | **Sum** | **450** |

    Covariance = 450 / (5-1) = 112.5 (We divide by n-1 for a sample). The positive number confirms they move together.

- Step 4: Calculate the Standard Deviation for Each Feature

    This measures the spread of each feature. The calculation involves squaring the deviations, averaging them, and taking the square root.
    - Standard Deviation of Temperature ≈ 7.91
    - Standard Deviation of Sales ≈ 14.36

- Step 5: Calculate the Correlation Coefficient

    Finally, divide the covariance by the product of the two standard deviations.

    Correlation=(Std Dev of Temp)×(Std Dev of Sales)Covariance​

    Correlation=(7.91×14.36)112.5​≈113.59112.5​≈0.99

This result of 0.99 is very close to +1, indicating a very strong positive correlation. In layman's terms, it's a near-perfect relationship: the hotter the day, the more ice cream you sell.

### Projection

A projection is the process of taking one object, like a point or a vector, and mapping it onto another object, like a line or a plane.1 In simple terms, it's about finding the "closest" point or the "shadow" of one object on another.2

**The Shadow Analogy 🔦**

The easiest way to understand projection is to think about shadows. Imagine you have a stick and a flat wall. You shine a flashlight directly at the wall from above.

- The stick is your original vector or object.
- The wall is the line or surface you are projecting onto.
- The shadow that the stick casts on the wall is its projection.

The shadow represents the "part" of the stick that lies along the direction of the wall. It shows you where the stick would land if you were to "flatten" it directly onto that surface.

**How It Works with Vectors**

In mathematics, especially in linear algebra and data science (like in Principal Component Analysis), we often project one vector onto another.

Imagine you have two vectors, v and u. You want to find the projection of vector v onto vector u.

This means you are trying to answer the question: "How much of vector v points in the same direction as vector u?"

The result is a new vector that:

1. Points in the same direction (or the exact opposite direction) as vector u.
2. Is the "component" of v that lies along u. It's the closest point on the line defined by vector u to the tip of vector v.

The projection essentially breaks down vector v into two parts: one part that is parallel to vector u (the projection itself) and another part that is perpendicular to it. This is incredibly useful for simplifying problems, reducing dimensions, and figuring out the components of a force or a data point in a specific direction.

### Change Of Axis

A "change of axis" in the context of projection is the process of switching from a standard coordinate system (like the default x and y axes) to a new set of axes that are better aligned with the structure of your data. This is done to simplify your data and make projections more meaningful.

Think of it as rotating your point of view to see the data's patterns more clearly. The goal is to make one of the new axes point in the direction of the data's most significant variation.

**The Map Analogy: Finding Your Way in a City 🗺️**

Imagine your data is a scatter plot of all the cafes in a city like Manhattan, where most of the activity happens along a long, diagonal avenue like Broadway.

1. The Original Axes (Standard View)
    - Your default map has two axes: one running East-West (x-axis) and another running North-South (y-axis).
    - To describe the location of a cafe, you need two coordinates: how far East it is and how far North it is.
    - If you wanted to "project" all the cafes onto the East-West axis to see their distribution, you'd get some information, but it wouldn't be very descriptive of the city's layout because you're ignoring the main flow of traffic along Broadway.
2. The Change of Axis (A Better View)
    - Now, what if you change your axes? You rotate your map so that your new "main axis" (let's call it the B-axis for Broadway) runs directly along Broadway.
    - Your new "secondary axis" (the C-axis for Cross-street) would run perpendicular to Broadway.
3. Projection in the New System (A More Meaningful Summary)
    - With this new system, you can project all the cafe locations onto your new B-axis. This projection is now incredibly useful. It tells you how far down the main avenue each cafe is located.
    - This single new coordinate captures the most important information about the cafes' locations. The second coordinate (the projection onto the C-axis) just tells you how far off the main avenue each cafe is, which is often less important information.
    - By doing this, you've essentially reduced the problem from two dimensions (East/North) to one dominant dimension (distance along Broadway) without losing the main story of the data.

**Connecting it to PCA and Data Science**

This is exactly what Principal Component Analysis (PCA) does.

1. Finding the New Axes: PCA analyzes the data to find the directions of maximum variance. The direction with the most spread (like Broadway in our example) becomes the First Principal Component (PC1). This is your new primary axis. The next most important direction, perpendicular to the first, is the Second Principal Component (PC2), and so on.
2. Changing the Axis: The process of finding these principal components is the "change of axis." You are defining a new coordinate system based on the shape of your data itself, not on some arbitrary x and y axes.
3. Projecting the Data: PCA then projects your original data points onto these new principal component axes. The coordinates of the data points in this new system tell you how much of each principal component is present in each data point. This projection is what allows for dimensionality reduction, as you can often describe your data very well using just the first few principal components and safely ignore the rest.

### Orthogonal Direction

In simple terms, orthogonal is the mathematical word for perpendicular. When two lines or vectors are orthogonal, they meet at a 90-degree angle.

**A Real-World Analogy: The Corner of a Room 部屋**

Look at the corner of any room.

- The line where two walls meet is orthogonal to the floor.
- The line where the floor meets one wall is orthogonal to the line where the floor meets the adjacent wall.

They all form right angles with each other. This concept of being at a right angle is what "orthogonal" means.

The concept of orthogonality is crucial in techniques like Principal Component Analysis (PCA) for a very specific reason: it ensures that the new axes (the principal components) are uncorrelated and independent.

When PCA performs a "change of axis," it finds a new set of axes to describe your data:
1. First Principal Component (PC1): It finds the direction with the most variance in the data.
2. Second Principal Component (PC2): It then finds the direction with the next most variance, with one critical rule: it must be orthogonal (perpendicular) to PC1.
3. Third Principal Component (PC3): This next axis must be orthogonal to both PC1 and PC2, and so on.

By making sure each new axis is orthogonal to the others, PCA guarantees that the information captured by each principal component is unique and not redundant. PC2 describes a pattern in the data that has nothing to do with the pattern described by PC1.

This makes the new coordinate system incredibly efficient and easy to interpret, as each axis provides a distinct piece of information about the data's structure.

### Dimensionality reduction

Dimensionality reduction is the process of reducing the number of variables (or "dimensions") in a dataset while trying to keep as much of the important information as possible. It's a way to simplify complex data, making it easier to process, visualize, and use in machine learning models.

Think of it as creating a high-quality summary of a large dataset by focusing on the most meaningful features.

**The Analogy: Describing a House for Sale 🏠**

Imagine you are a real estate agent trying to describe a house. You could list every single detail:

- Number of bedrooms
- Number of bathrooms
- Square footage
- Age of the house
- Size of the backyard
- Number of windows
- Color of the front door
- Brand of the kitchen appliances
- Type of flooring in the living room
- ...and 50 other features.

This is a high-dimensional dataset. A potential buyer might get overwhelmed by all these details.

To simplify, you realize that some features can be combined. For instance, square footage, number of bedrooms, and number of bathrooms all relate to the overall "Size" of the house. Similarly, age of the house, brand of appliances, and type of flooring could be combined into a new, more intuitive feature called "Modernity".

Instead of using 50+ variables, you can now describe the house very effectively with just a few key dimensions like Size, Modernity, and Location. You've reduced the dimensionality of your data, making it much easier to understand and compare different houses without losing the essential information.

**Why is Dimensionality Reduction Important?**

1. Easier Visualization: Humans can't visualize data in more than three dimensions. By reducing data to two or three dimensions, you can create scatter plots to see patterns, clusters, and outliers.
2. Faster Model Training: Fewer variables mean less data for a machine learning algorithm to process, which leads to significantly faster training times.
3. Avoiding the "Curse of Dimensionality": When you have too many features compared to the number of data points, models can become too complex and start to "memorize" the training data instead of learning the underlying patterns. This leads to poor performance on new, unseen data. Reducing dimensions helps prevent this.
4. Less Redundancy: Many datasets have redundant features (features that are highly correlated and provide similar information). Dimensionality reduction helps to remove this redundancy.

**Approaches**

There are two primary ways to reduce the dimensions of your data:

1. Feature Selection
    
    This is the simpler approach. You analyze all your existing variables and simply select a subset of the most important ones, discarding the rest. It's like deciding that out of your 50 house features, only square footage, location, age, and backyard size really matter for the price, so you just keep those.
    - Pros: Easy to understand and the resulting features are still the original ones (e.g., "square footage").
    - Cons: You might lose some information contained in the discarded variables.
2. Feature Extraction

    This is a more advanced approach. Instead of just selecting from existing features, you create new, combined features from the old ones. This is what we did in the house analogy by creating "Size" and "Modernity."

    The most common technique for this is Principal Component Analysis (PCA). PCA finds the underlying patterns in the data and creates new, uncorrelated variables called principal components. The first few principal components capture the vast majority of the information, so you can keep them and discard the rest.

    - Pros: It often retains more information than feature selection because it combines and condenses the original variables.
    - Cons: The new features are combinations of the old ones and may not have a clear, real-world meaning, making them harder to interpret.


### Autocorrelation Matrix

An autocorrelation matrix is a square table that shows the autocorrelation of a signal or a time series with itself at different time lags. In simple terms, it's a way to see how a variable's past values relate to its current value.

This matrix is a fundamental tool in signal processing and time-series analysis, helping to identify repeating patterns or dependencies over time.

**The Core Idea: Self-Comparison at Different Lags**

Before understanding the matrix, you need to understand autocorrelation. Autocorrelation means "self-correlation". It measures the similarity between a time series and a delayed version of itself.

- A lag of 0 means you are correlating the series with itself perfectly, which always results in a correlation of 1.
- A lag of 1 means you are comparing the series's value at a given time t with its value at the previous time t-1.
- A lag of 2 means you are comparing the value at time t with the value at time t-2, and so on.

**Structure of the Matrix**

An autocorrelation matrix organizes these lag correlations in a simple grid. For a signal with p lags, the matrix will be a p x p grid.

The element in the i-th row and j-th column of the matrix shows the correlation between the signal at lag i and lag j.

Let's say you are looking at the daily temperature and you want to see the autocorrelation for lags of 0, 1, and 2 days. The matrix would look like this:

$$
R = \begin{bmatrix}
\text{Corr(lag 0, lag 0)} & \text{Corr(lag 0, lag 1)} & \text{Corr(lag 0, lag 2)} \\
\text{Corr(lag 1, lag 0)} & \text{Corr(lag 1, lag 1)} & \text{Corr(lag 1, lag 2)} \\
\text{Corr(lag 2, lag 0)} & \text{Corr(lag 2, lag 1)} & \text{Corr(lag 2, lag 2)}
\end{bmatrix}
$$

**Key Properties**

Autocorrelation matrices have some important and simplifying properties.

1. Ones on the Diagonal: The correlation of any lag with itself is always perfect, so the main diagonal is always filled with 1s.
2. Symmetric: The correlation between lag 1 and lag 2 is the same as the correlation between lag 2 and lag 1. This means the matrix is symmetric across its main diagonal.

Let's look at a real example. Suppose we calculated the following autocorrelations for our daily temperature data:

- Correlation at lag 0 = 1.0
- Correlation at lag 1 (today vs. yesterday) = 0.8
- Correlation at lag 2 (today vs. the day before yesterday) = 0.6

The resulting 3x3 autocorrelation matrix would be:

$$
R = \begin{bmatrix}
1.0 & 0.8 & 0.6 \\
0.8 & 1.0 & 0.8 \\
0.6 & 0.8 & 1.0
\end{bmatrix}
$$

This matrix tells us:

- Today's temperature is strongly correlated with yesterday's temperature (0.8).
- It's also correlated with the temperature from two days ago, but a bit less so (0.6).

This information is very useful for forecasting, as it shows that past temperatures are a good predictor of future temperatures.

### Eigenvectors - Symmetric Matrices 

An eigenvector of a matrix is a special non-zero vector that, when the matrix acts on it, does not change its direction but is only scaled by a certain value. For symmetric matrices, these eigenvectors have a particularly important and useful property: they are always orthogonal to each other.

**Eigenvectors and Eigenvalues Explained**

Imagine a matrix as a transformation that can stretch, shrink, or rotate vectors in space.

- An eigenvector is a vector that, after being transformed by the matrix, still points in the same (or exact opposite) direction. The "eigen" prefix is German for "own" or "characteristic," so it's a vector that maintains its characteristic direction under the transformation.
- The eigenvalue is the scalar factor by which the eigenvector is stretched or shrunk. A positive eigenvalue means it stretches in the same direction, while a negative eigenvalue means it flips and stretches in the opposite direction.

The relationship is defined by the equation: Av=λv

Where:

- A is the matrix.
- v is the eigenvector.
- λ (lambda) is the eigenvalue.


#### Symmetric Matrices

A symmetric matrix is a square matrix that is equal to its transpose. In simple terms, if you flip the matrix across its main diagonal (from top-left to bottom-right), it looks exactly the same.

Example of a Symmetric Matrix:

A = ​521​234​146​

Notice that the element at row i, column j is the same as the element at row j, column i.

**The Special Property of Eigenvectors for Symmetric Matrices**

When a matrix is symmetric, its eigenvectors corresponding to distinct eigenvalues have a crucial property: they are orthogonal. This means they are perpendicular to each other, meeting at a 90-degree angle.

Think back to the standard x, y, and z axes in 3D space. They are all mutually orthogonal. The eigenvectors of a symmetric matrix form a similar set of perpendicular axes that are perfectly aligned with the directions of the matrix's transformation.

Because the eigenvectors of a symmetric matrix are orthogonal, they can form a new, ideal coordinate system for the data or transformation represented by that matrix. This is the fundamental concept that makes techniques like Principal Component Analysis (PCA) so powerful.

In PCA, the covariance matrix (which is always symmetric) is analyzed. Its eigenvectors point in the directions of the most variance in the data (the principal components), and because they are orthogonal, they represent uncorrelated, independent patterns. This allows you to rotate the dataset to this new, more informative set of axes without losing any information.

### Five Steps - PCA

Principal Component Analysis (PCA) is a technique for reducing the number of variables in a dataset while retaining most of the original information. It achieves this by creating new, uncorrelated variables called principal components.

To illustrate, we'll use a very simple dataset of student scores in two subjects: Math and Physics. Our goal is to reduce these two dimensions to just one.

| Student | Math (X) | Physics (Y) |
|---|---|---|
| A | 90 | 85 |
| B | 70 | 60 |
| C | 95 | 90 |
| D | 60 | 50 |

1. Step 1: Standardize the Data

    Standardization rescales the data to have a mean of 0 and a standard deviation of 1. This is crucial because PCA is sensitive to the scale of the variables. If one variable has a much larger range than another (e.g., salary in thousands vs. age in years), it will dominate the analysis.

    1. Calculate the mean for each variable.
        - Mean of Math: (90 + 70 + 95 + 60) / 4 = 78.75
        - Mean of Physics: (85 + 60 + 90 + 50) / 4 = 71.25
    2. Calculate the standard deviation for each variable.
        - Standard Deviation of Math ≈ 16.5
        - Standard Deviation of Physics ≈ 18.9
    3. For each data point, use the formula: (Value - Mean) / Standard Deviation.

    After standardization, our data looks something like this:


    | Student | Math (Standardized) | Physics (Standardized) |
    |---|---|---|
    | A | 0.68 | 0.73 |
    | B | -0.53 | -0.60 |
    | C | 0.98 | 0.99 |
    | D | -1.14 | -1.12 |

2. Step 2: Compute the Covariance Matrix

    The covariance matrix is a square table that shows how the different variables in the dataset vary with each other. A positive value means they tend to increase together, while a negative value means one tends to increase as the other decreases.

    For our 2D data, we'll get a 2x2 matrix:

    $$
    \begin{bmatrix}
    \text{Var(Math)} & \text{Cov(Math, Physics)} \\
    \text{Cov(Physics, Math)} & \text{Var(Physics)}
    \end{bmatrix}
    $$

    Since our data is standardized, the variance of each variable is 1. The covariance between Math and Physics turns out to be very high (let's say 0.99 for this example, as they are highly correlated).
    Our covariance matrix is:

    $$
    \begin{bmatrix}
    1.0 & 0.99 \\
    0.99 & 1.0
    \end{bmatrix}
    $$

3. Step 3: Compute Eigenvectors and Eigenvalues

    This is the core of PCA. We decompose the covariance matrix to find its eigenvectors and eigenvalues.

    - Eigenvectors are the directions of the new axes (the principal components) where the data has the most variance.
    - Eigenvalues are numbers that tell you the amount of variance captured by each eigenvector.

    Through linear algebra, we calculate the eigenvectors and eigenvalues of the covariance matrix. For our example, we would get two eigenvectors and two corresponding eigenvalues:

    - Eigenvalue 1 = 1.99 (This is the larger one)
        - Eigenvector 1 = [0.707, 0.707]
    - Eigenvalue 2 = 0.01
        - Eigenvector 2 = [-0.707, 0.707]
    
    The first eigenvector [0.707, 0.707] points in the direction of the greatest variance in the data. This will be our first principal component.

4. Step 4: Choose the Principal Components

    We rank the eigenvectors by their corresponding eigenvalues in descending order. The eigenvector with the highest eigenvalue is the most significant and becomes the first principal component (PC1). We then decide how many components to keep.

    - Our first eigenvalue (1.99) is much larger than the second (0.01).
    - The total variance is the sum of the eigenvalues: 1.99 + 0.01 = 2.0.
    - The first component explains 1.99 / 2.0 = 99.5% of the total variance.
    - The second component only explains 0.01 / 2.0 = 0.5% of the variance.

    Since PC1 captures almost all the information, we can safely discard PC2 and reduce our data from two dimensions to just one. We choose Eigenvector 1 to be our single new axis.

5. Step 5: Project the Data onto the New Axis

    The final step is to transform our original, standardized data into the new coordinate system defined by the principal components we chose. This is done by projecting the data points onto the new axis (our chosen eigenvector).

    We take the dot product of our standardized data and the chosen eigenvector ([0.707, 0.707]).

    | Student | Standardized Data | PC1 (New Single Dimension) |
    |---|---|---|
    | A | [0.68, 0.73] | (0.68*0.707) + (0.73*0.707) = 0.99 |
    | B | [-0.53, -0.60] | (-0.53*0.707) + (-0.60*0.707) = -0.80 |
    | C | [0.98, 0.99] | (0.98*0.707) + (0.99*0.707) = 1.39 |
    | D | [-1.14, -1.12] | (-1.14*0.707) + (-1.12*0.707) = -1.59 |

    We have now successfully reduced our two-dimensional dataset (Math and Physics scores) to a single, highly informative dimension (PC1), which we could call "Overall Academic Performance."


### Quiz --> [Principal Component Analyses Quiz](./Quiz/PrincipalComponentAnalysesQuiz.md)

### Previous Topic --> [Parameter Estimation](./ParameterEstimation.md)
### Next Topic --> [Non-Parametric Techniques](./Non-ParametricTechnique.md)
</div>