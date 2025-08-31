# Non-Parametric Techniques Quiz

Here are 50 multiple-choice questions based on the Non-Parametric Techniques notes.

### Questions

#### General Concepts (Parametric vs. Non-Parametric)

1.  **What is the primary goal of density estimation?**
    *   [ ] A) To classify data into predefined categories.
    *   [ ] B) To create a smooth curve or shape that shows where data points are most concentrated.
    *   [ ] C) To select a random sample from a population.
    *   [ ] D) To calculate the average of a dataset.

    **Answer: B) To create a smooth curve or shape that shows where data points are most concentrated.**

    **Explanation:**
    *   A is incorrect. This is classification.
    *   B is correct. Density estimation aims to model the underlying probability distribution from which a set of data is drawn, showing the "density" of data across the range of values.
    *   C is incorrect. This is sampling.
    *   D is incorrect. This is calculating a statistic, not estimating a distribution.

2.  **Which of the following best describes a parametric distribution?**
    *   [ ] A) A distribution that always has a U-shape.
    *   [ ] B) A distribution that can only be described by a histogram.
    *   [ ] C) A distribution whose shape is completely defined by a fixed, finite number of parameters.
    *   [ ] D) A distribution whose shape is determined by the data itself, without any assumptions.

    **Answer: C) A distribution whose shape is completely defined by a fixed, finite number of parameters.**

    **Explanation:**
    *   A is incorrect. A U-shape is just one of many possible distribution shapes.
    *   B is incorrect. Histograms are a non-parametric tool.
    *   C is correct. In a parametric model, if you know the parameters (e.g., mean and standard deviation for a Normal distribution), you know everything about the distribution's shape.
    *   D describes a non-parametric approach.

3.  **What is the critical first step in parametric density estimation?**
    *   [ ] A) Finding the nearest neighbors for each data point.
    *   [ ] B) Choosing a specific probability distribution model that you assume the data follows.
    *   [ ] C) Creating a histogram of the data.
    *   [ ] D) Calculating the mean of the data.

    **Answer: B) Choosing a specific probability distribution model that you assume the data follows.**

    **Explanation:**
    *   A is a non-parametric technique.
    *   B is correct. The core of the parametric approach is making an upfront assumption about the data's underlying distribution (e.g., assuming it is Normal).
    *   C is a non-parametric technique.
    *   D is part of the process but not the first step.

4.  **Which of these is a key characteristic of non-parametric estimation?**
    *   [ ] A) It is always computationally faster than parametric methods.
    *   [ ] B) It makes strong assumptions about the shape of the data distribution.
    *   [ ] C) It lets the data speak for itself, building a model whose complexity can grow with the data.
    *   [ ] D) It requires a small, fixed number of parameters.

    **Answer: C) It lets the data speak for itself, building a model whose complexity can grow with the data.**

    **Explanation:**
    *   A is incorrect. Non-parametric methods are often more computationally intensive.
    *   B and D are characteristics of parametric estimation.
    *   C is correct. Non-parametric methods are flexible and data-driven; they don't try to force the data into a preconceived shape.

5.  **The Poisson distribution is a parametric distribution defined by what single parameter?**
    *   [ ] A) The average rate (λ).
    *   [ ] B) The probability of success (p).
    *   [ ] C) The standard deviation (σ).
    *   [ ] D) The number of trials (n).

    **Answer: A) The average rate (λ).**

    **Explanation:**
    *   A is correct. The Poisson distribution is defined by a single parameter, λ (lambda), which represents the average number of events in a fixed interval.
    *   B and D are parameters for the Binomial distribution.
    *   C is a parameter for the Normal distribution.

6.  **Using a "paint-by-numbers" kit is an analogy for which type of estimation?**
    *   [ ] A) Parametric estimation
    *   [ ] B) Nearest neighbor estimation
    *   [ ] C) Histogram-based estimation
    *   [ ] D) Non-parametric estimation

    **Answer: A) Parametric estimation**

    **Explanation:**
    *   The analogy fits because in parametric estimation, the structure (the pre-drawn lines) is assumed beforehand, and the only task is to find the parameters (fill in the colors).

7.  **What is a major advantage of parametric density estimation?**
    *   [ ] A) It does not require any data to work.
    *   [ ] B) It is efficient and can be described by a simple mathematical formula.
    *   [ ] C) It never makes assumptions about the data.
    *   [ ] D) It is flexible and can fit any data shape.

    **Answer: B) It is efficient and can be described by a simple mathematical formula.**

    **Explanation:**
    *   A is incorrect.
    *   B is correct. Once the parameters are estimated, the model is a compact and efficient mathematical function that is easy to work with.
    *   C and D are incorrect; these are features of non-parametric methods.

8.  **A U-shaped dataset is an example of a:**
    *   [ ] A) Poisson distribution
    *   [ ] B) Normal distribution
    *   [ ] C) Bimodal distribution
    *   [ ] D) Unimodal distribution

    **Answer: C) Bimodal distribution**

    **Explanation:**
    *   A U-shaped distribution has two peaks (modes) at the extremes of the data range, making it a specific type of bimodal distribution.

9.  **If you use a parametric method on data that does not fit the assumed distribution, what is the likely outcome?**
    *   [ ] A) The model will have very high variance.
    *   [ ] B) The model will be a poor representation of the data's true density.
    *   [ ] C) The model will automatically switch to a non-parametric method.
    *   [ ] D) The model will be a perfect fit.

    **Answer: B) The model will be a poor representation of the data's true density.**

    **Explanation:**
    *   The biggest risk of parametric methods is that if the initial assumption about the distribution's shape is wrong, the resulting model will be biased and will not accurately reflect the data.

10. **The Binomial distribution would be most appropriate for which of the following scenarios?**
    *   [ ] A) Modeling customer satisfaction scores on a 1-5 star scale.
    *   [ ] B) Modeling the number of emails arriving in an inbox per hour.
    *   [ ] C) Modeling the outcome of flipping a coin 10 times.
    *   [ ] D) Modeling the heights of adults.

    **Answer: C) Modeling the outcome of flipping a coin 10 times.**

    **Explanation:**
    *   A might be modeled by a U-shaped or other distribution.
    *   B is a classic example of a Poisson distribution.
    *   C is correct. The Binomial distribution models the number of successes (e.g., heads) in a fixed number of independent binary trials (the coin flips).
    *   D is typically modeled by a Normal distribution.

#### Histogram Methods

11. **Histogram-based density estimation is considered non-parametric because:**
    *   [ ] A) It is computationally very simple.
    *   [ ] B) It does not assume any underlying shape for the data distribution.
    *   [ ] C) The total area of the bars always sums to 1.
    *   [ ] D) It uses a fixed number of bins.

    **Answer: B) It does not assume any underlying shape for the data distribution.**

    **Explanation:**
    *   The shape of the histogram is determined entirely by the data and the choice of bins, not by a pre-selected mathematical function like a bell curve.

12. **What is the most crucial parameter to choose when creating a histogram?**
    *   [ ] A) The height of the y-axis.
    *   [ ] B) The total number of data points.
    *   [ ] C) The bin width.
    *   [ ] D) The color of the bars.

    **Answer: C) The bin width.**

    **Explanation:**
    *   The bin width controls the level of smoothing and can drastically change the appearance and interpretation of the histogram.

13. **If you choose a bin width that is too wide, what is the likely result?**
    *   [ ] A) A histogram with very low bias and high variance.
    *   [ ] B) A very smooth histogram that hides important details (high bias).
    *   [ ] C) A histogram where each data point gets its own bin.
    *   [ ] D) A very noisy and jagged histogram.

    **Answer: B) A very smooth histogram that hides important details (high bias).**

    **Explanation:**
    *   Wide bins lump too much data together, oversimplifying the distribution and potentially hiding features like multiple peaks. This is a high-bias model.

14. **If you choose a bin width that is too narrow, what is the likely result?**
    *   [ ] A) A histogram with very high bias and low variance.
    *   [ ] B) A histogram that is easy to interpret.
    *   [ ] C) A very noisy and jagged histogram that follows the data too closely (high variance).
    *   [ ] D) A very smooth, generalized histogram.

    **Answer: C) A very noisy and jagged histogram that follows the data too closely (high variance).**

    **Explanation:**
    *   Narrow bins make the histogram highly sensitive to random fluctuations in the data, creating a spiky shape that is hard to interpret. This is a high-variance model.

15. **The choice of bin width is a classic example of the:**
    *   [ ] A) Central Limit Theorem.
    *   [ ] B) Bias-variance trade-off.
    *   [ ] C) Curse of dimensionality.
    *   [ ] D) Parametric vs. Non-parametric dilemma.

    **Answer: B) The bias-variance trade-off.**

    **Explanation:**
    *   Wide bins lead to high bias and low variance, while narrow bins lead to low bias and high variance. The goal is to find a balance between the two.

16. **In a normalized histogram, what must the total area of all bars sum to?**
    *   [ ] A) 1
    *   [ ] B) The total number of samples.
    *   [ ] C) 100
    *   [ ] D) The number of bins.

    **Answer: A) 1**

    **Explanation:**
    *   For the histogram to represent a probability density, the total area must equal 1, signifying that the total probability of all outcomes is 100%.

17. **The formula for density in a histogram is (Frequency) / (Total Samples * X). What is X?**
    *   [ ] A) The standard deviation of the data.
    *   [ ] B) The bin width.
    *   [ ] C) The mean of the data.
    *   [ ] D) The number of bins.

    **Answer: B) The bin width.**

    **Explanation:**
    *   The height of each bar is the count (frequency) divided by the total number of samples (to get a proportion) and then divided by the bin width (to make the area, not the height, represent the proportion).

18. **Looking at a city from a satellite is an analogy for using:**
    *   [ ] A) A kernel density estimate.
    *   [ ] B) Very wide bins.
    *   [ ] C) A parametric model.
    *   [ ] D) Very narrow bins.

    **Answer: B) Very wide bins.**

    **Explanation:**
    *   From a satellite, you see the overall shape (a smooth, low-detail view) but miss the fine details, just like a histogram with very wide bins.

19. **A major disadvantage of histograms for density estimation is that the resulting shape is:**
    *   [ ] A) Always bell-shaped.
    *   [ ] B) Not dependent on the start and end points of the bins.
    *   [ ] C) Discontinuous and blocky.
    *   [ ] D) Always smooth and continuous.

    **Answer: C) Discontinuous and blocky.**

    **Explanation:**
    *   Histograms have sharp edges at the bin boundaries, which is often not a natural representation of an underlying distribution. This discontinuity is a key limitation.

20. **The bin width in a histogram is analogous to what parameter in Kernel Density Estimation?**
    *   [ ] A) The final density estimate (f(x))
    *   [ ] B) The number of samples (n)
    *   [ ] C) The smoothing parameter or bandwidth (h)
    *   [ ] D) The kernel function (K)

    **Answer: C) The smoothing parameter or bandwidth (h)**

    **Explanation:**
    *   Both the bin width and the bandwidth (h) control the level of smoothing in their respective methods and involve the same bias-variance trade-off.

#### Kernel Density Estimation (KDE) / Parzen Window

21. **What is Kernel Density Estimation (KDE)?**
    *   [ ] A) A method for calculating the mean and standard deviation.
    *   [ ] B) A parametric method for fitting a bell curve to data.
    *   [ ] C) A non-parametric method for creating a smooth, continuous density curve from data.
    *   [ ] D) A method for creating a blocky histogram with wide bins.

    **Answer: C) A non-parametric method for creating a smooth, continuous density curve from data.**

    **Explanation:**
    *   KDE (also known as the Parzen window method) is a sophisticated technique that avoids the blocky nature of histograms by summing smooth kernel functions to produce a smooth density estimate.

22. **In the "pebbles in a pond" analogy for KDE, what does each pebble represent?**
    *   [ ] A) The final density curve
    *   [ ] B) A data point
    *   [ ] C) The bandwidth
    *   [ ] D) A bin

    **Answer: B) A data point**

    **Explanation:**
    *   The core idea of KDE is to place a kernel (a ripple) at the location of every single data point (a pebble).

23. **In KDE, what is the "kernel"?**
    *   [ ] A) The number of data points.
    *   [ ] B) The raw dataset.
    *   [ ] C) A smooth "bump" or weighting function placed on each data point.
    *   [ ] D) The final, overall density curve.

    **Answer: C) A smooth "bump" or weighting function placed on each data point.**

    **Explanation:**
    *   The kernel is a function that defines the shape of the influence each data point has on its surroundings. The final KDE curve is the sum of all these individual bumps.

24. **The Parzen window method is another name for:**
    *   [ ] A) The k-Nearest Neighbors method
    *   [ ] B) Parametric estimation
    *   [ ] C) Kernel Density Estimation (KDE)
    *   [ ] D) Histogram-based estimation

    **Answer: C) Kernel Density Estimation (KDE)**

    **Explanation:**
    *   The Parzen-Rosenblatt window method is the formal name for what is commonly known as Kernel Density Estimation.

25. **What is the primary role of the bandwidth (h) in the KDE formula?**
    *   [ ] A) It ensures the area under the curve is exactly 1.
    *   [ ] B) It controls the number of bumps.
    *   [ ] C) It controls the width (smoothing) of the bumps.
    *   [ ] D) It controls the height of the bumps.

    **Answer: C) It controls the width (smoothing) of the bumps.**

    **Explanation:**
    *   The bandwidth `h` is the critical smoothing parameter. A small `h` leads to narrow, spiky kernels (high variance), while a large `h` leads to wide, overly smooth kernels (high bias).

26. **Which type of kernel gives equal weight to all points within its bandwidth and zero weight outside?**
    *   [ ] A) All of the above
    *   [ ] B) Triangular kernel
    *   [ ] C) The box kernel (or rectangular kernel)
    *   [ ] D) Gaussian kernel

    **Answer: C) The box kernel (or rectangular kernel)**

    **Explanation:**
    *   The box kernel is the simplest, defined by a rectangular window that treats all points inside the window equally, resulting in a blocky, step-like density estimate.

27. **What is the defining characteristic of a Gaussian kernel?**
    *   [ ] A) It can only be used for data that is normally distributed.
    *   [ ] B) It assigns weight that diminishes smoothly and symmetrically in a "bell curve" shape.
    *   [ ] C) It gives equal weight to all points within its bandwidth.
    *   [ ] D) It creates a triangular-shaped bump.

    **Answer: B) It assigns weight that diminishes smoothly and symmetrically in a "bell curve" shape.**

    **Explanation:**
    *   The Gaussian kernel uses the formula for a Normal distribution to assign weights, meaning points closer to the center have a much higher influence than points farther away, with the influence dropping off smoothly.

28. **What is the most significant disadvantage of Kernel Density Estimation?**
    *   [ ] A) It cannot be used for datasets with more than 100 points.
    *   [ ] B) It is a parametric method with strong assumptions.
    *   [ ] C) Its result is critically dependent on the choice of bandwidth (h).
    *   [ ] D) It produces a blocky, discontinuous curve.

    **Answer: C) Its result is critically dependent on the choice of bandwidth (h).**

    **Explanation:**
    *   While KDE has many advantages, its biggest weakness is its sensitivity to the bandwidth parameter. An incorrect choice of `h` can lead to a very misleading density curve.

29. **In the KDE formula, what does K((x - xi) / h) represent?**
    *   [ ] A) The total area under the curve.
    *   [ ] B) The estimated density at point x.
    *   [ ] C) The contribution of a single data point (xi) to the density at point x.
    *   [ ] D) The total number of data points.

    **Answer: C) The contribution of a single data point (xi) to the density at point x.**

    **Explanation:**
    *   This term calculates the value of the kernel function based on the scaled distance between the evaluation point `x` and a specific data point `xi`.

30. **Compared to a histogram, a major advantage of KDE is that it:**
    *   [ ] A) Is less sensitive to the data.
    *   [ ] B) Produces a smooth and continuous curve.
    *   [ ] C) Does not require any parameters to be chosen.
    *   [ ] D) Is less computationally intensive.

    **Answer: B) Produces a smooth and continuous curve.**

    **Explanation:**
    *   The primary benefit of KDE over a histogram is its ability to produce a smooth, continuous density estimate, which is often a more natural and interpretable representation of the underlying distribution.

#### Nearest Neighbour Methods

31. **What is the fundamental difference between the Parzen window (KDE) and the Nearest Neighbor method for density estimation?**
    *   [ ] A) Parzen window is only for classification, while Nearest Neighbor is only for density estimation.
    *   [ ] B) Parzen window uses a fixed volume and counts the points inside, while Nearest Neighbor uses a fixed number of points and measures the volume.
    *   [ ] C) Parzen window uses a fixed number of points (k), while Nearest Neighbor uses a fixed bandwidth (h).
    *   [ ] D) Parzen window is parametric, while Nearest Neighbor is non-parametric.

    **Answer: B) Parzen window uses a fixed volume and counts the points inside, while Nearest Neighbor uses a fixed number of points and measures the volume.**

    **Explanation:**
    *   This is the key distinction. KDE/Parzen has a fixed bandwidth `h` (which defines a fixed volume) and the number of points `k` inside that volume varies. The Nearest Neighbor method has a fixed `k` and the volume `V` containing those `k` points varies.

32. **In the Nearest Neighbor density estimation formula, f(x) = k / (n * V), what does V represent?**
    *   [ ] A) The total number of data points.
    *   [ ] B) The value of the k-th neighbor.
    *   [ ] C) The volume of the sphere containing the k nearest neighbors.
    *   [ ] D) The variance of the data.

    **Answer: C) The volume of the sphere containing the k nearest neighbors.**

    **Explanation:**
    *   The density is calculated as the number of points (`k`) divided by the total volume (`V`) they occupy (scaled by `n`).

33. **In the Nearest Neighbor method, if the data points in a region are tightly packed, the volume V will be:**
    *   [ ] A) Small, resulting in a low density estimate.
    *   [ ] B) Small, resulting in a high density estimate.
    *   [ ] C) Large, resulting in a low density estimate.
    *   [ ] D) Large, resulting in a high density estimate.

    **Answer: B) Small, resulting in a high density estimate.**

    **Explanation:**
    *   If points are close together, the volume `V` needed to enclose `k` of them will be small. Since `V` is in the denominator of the density formula (k/nV), a small `V` leads to a large (high) density estimate.

34. **What is the smoothing parameter in the Nearest Neighbor method?**
    *   [ ] A) The total number of samples (n)
    *   [ ] B) The volume (V)
    *   [ ] C) The number of neighbors (k)
    *   [ ] D) The bandwidth (h)

    **Answer: C) The number of neighbors (k)**

    **Explanation:**
    *   `k` controls the level of smoothing. A small `k` leads to a noisy estimate, while a large `k` leads to a smoother estimate.

35. **Using a small value for k in the k-NN algorithm leads to:**
    *   [ ] A) A model that is not sensitive to individual data points.
    *   [ ] B) A complex, jagged decision boundary that is sensitive to noise (high variance).
    *   [ ] C) Underfitting.
    *   [ ] D) A smooth decision boundary (high bias).

    **Answer: B) A complex, jagged decision boundary that is sensitive to noise (high variance).**

    **Explanation:**
    *   A small `k` (like k=1) means the classification is based on only a few local points, making the decision boundary very flexible and prone to following the noise in the training data. This is a low-bias, high-variance model.

36. **Using a very large value for k in the k-NN algorithm leads to:**
    *   [ ] A) A model with low bias and high variance.
    *   [ ] B) Overfitting.
    *   [ ] C) A very smooth decision boundary that might miss local patterns (high bias).
    *   [ ] D) A complex decision boundary.

    **Answer: C) A very smooth decision boundary that might miss local patterns (high bias).**

    **Explanation:**
    *   A large `k` averages the decision over a large neighborhood, smoothing out the decision boundary. This makes the model more stable but can cause it to miss the finer structure in the data (underfitting). This is a high-bias, low-variance model.

37. **What is the k=1 Nearest Neighbor rule?**
    *   [ ] A) A new point is assigned a random label.
    *   [ ] B) A new point is assigned the label of its single closest neighbor.
    *   [ ] C) A new point is assigned the most common label in the entire dataset.
    *   [ ] D) A new point is assigned the average label of all data points.

    **Answer: B) A new point is assigned the label of its single closest neighbor.**

    **Explanation:**
    *   This is the simplest case of the k-NN algorithm, where the decision is based entirely on the class of the one data point that is nearest to the new point.

38. **The decision boundary created by the k=1 NN rule is a mosaic of:**
    *   [ ] A) Bell curves
    *   [ ] B) Triangles
    *   [ ] C) Voronoi cells
    *   [ ] D) Circles

    **Answer: C) Voronoi cells**

    **Explanation:**
    *   The k=1 rule partitions the feature space into regions (Voronoi cells), where each region consists of all points closer to one particular training sample than to any other.

39. **In k-NN classification, how is the label for a new point determined?**
    *   [ ] A) By using a parametric formula.
    *   [ ] B) By taking a majority vote among the labels of the k nearest neighbors.
    *   [ ] C) By calculating the average of the labels.
    *   [ ] D) By the label of the farthest neighbor.

    **Answer: B) By taking a majority vote among the labels of the k nearest neighbors.**

    **Explanation:**
    *   For classification, the algorithm identifies the `k` nearest neighbors and assigns the new point the class label that is most common among those neighbors.

40. **The choice of k in k-NN is a classic example of:**
    *   [ ] A) Density estimation.
    *   [ ] B) The bias-variance trade-off.
    *   [ ] C) Parametric estimation.
    *   [ ] D) The curse of dimensionality.

    **Answer: B) The bias-variance trade-off.**

    **Explanation:**
    *   Just like the bin width in histograms and bandwidth in KDE, `k` controls the model's complexity. A small `k` has low bias but high variance, while a large `k` has high bias but low variance.

41. **What is a major drawback of the Nearest Neighbor method for density estimation?**
    *   [ ] A) It is faster than KDE for all datasets.
    *   [ ] B) The method is parametric.
    *   [ ] C) The estimated density function is not a true probability density because it may not integrate to 1.
    *   [ ] D) The resulting density function is always smooth.

    **Answer: C) The estimated density function is not a true probability density because it may not integrate to 1.**

    **Explanation:**
    *   A known issue with this method is that the resulting function f(x) does not necessarily integrate to 1 over its entire domain, which is a requirement for a true probability density function. The resulting curve can also be very spiky.

42. **The k-NN algorithm is considered a "lazy learner" because:**
    *   [ ] A) It can only be used for small datasets.
    *   [ ] B) It does no real "learning" during the training phase and simply stores the entire dataset.
    *   [ ] C) It is computationally inefficient.
    *   [ ] D) It does all the work during the training phase.

    **Answer: B) It does no real "learning" during the training phase and simply stores the entire dataset.**

    **Explanation:**
    *   Unlike models like linear regression that learn a function during training, k-NN does nothing until a new point needs to be classified. All the work (calculating distances) happens at prediction time.

43. **What does a U-shaped distribution suggest about the data?**
    *   [ ] A) The data follows a Normal distribution.
    *   [ ] B) Most data points are clustered in the middle of the range.
    *   [ ] C) The data is polarized into two distinct, opposing groups.
    *   [ ] D) The data is uniformly distributed.

    **Answer: C) The data is polarized into two distinct, opposing groups.**

    **Explanation:**
    *   The high frequencies at the extremes and low frequency in the middle indicate that the data points tend to fall into one of two separate groups.

44. **Why can the mean be a misleading statistic for a U-shaped dataset?**
    *   [ ] A) Because the mean will be the same as the mode.
    *   [ ] B) Because the mean will fall in the center of the range, where the data is scarcest.
    *   [ ] C) Because a U-shaped dataset cannot have a mean.
    *   [ ] D) Because the mean will be at one of the two peaks.

    **Answer: B) Because the mean will fall in the center of the range, where the data is scarcest.**

    **Explanation:**
    *   The mean, as a measure of central tendency, will be located in the "dip" of the U, a value that is not at all typical of the data points, which are clustered at the ends.

45. **Which of these is NOT a non-parametric technique?**
    *   [ ] A) Estimating the density of data by assuming it follows a Normal distribution.
    *   [ ] B) k-Nearest Neighbors (k-NN).
    *   [ ] C) Kernel Density Estimation (KDE).
    *   [ ] D) Histogram-based density estimation.

    **Answer: A) Estimating the density of data by assuming it follows a Normal distribution.**

    **Explanation:**
    *   Assuming the data follows a specific distribution like the Normal distribution is the definition of a parametric technique.

46. **The "kernel trick" is a mathematical shortcut used to:**
    *   [ ] A) Calculate the mean of a dataset.
    *   [ ] B) Choose the optimal number of bins for a histogram.
    *   [ ] C) Smooth the decision boundary of a k-NN model.
    *   [ ] D) Calculate the similarity between points in a higher dimension without transforming them.

    **Answer: D) Calculate the similarity between points in a higher dimension without transforming them.**

    **Explanation:**
    *   The kernel trick is a core concept in machine learning (especially SVMs) where a kernel function computes the dot product of data points in a higher-dimensional feature space without explicitly mapping the points to that space, saving computation.

47. **Which kernel function gives linearly decreasing weight from the center to the edge of the window?**
    *   [ ] A) All of the above
    *   [ ] B) Gaussian kernel
    *   [ ] C) Triangular kernel
    *   [ ] D) Box kernel

    **Answer: C) Triangular kernel**

    **Explanation:**
    *   The shape of the triangular kernel is a triangle, meaning the influence of points decreases linearly with distance from the center, unlike the flat box kernel or the bell-shaped Gaussian kernel.

48. **A key drawback of KDE is that a Gaussian kernel can estimate non-zero density for impossible values (e.g., negative house prices). Why does this happen?**
    *   [ ] A) Because KDE is a parametric method.
    *   [ ] B) Because the bandwidth `h` was chosen incorrectly.
    *   [ ] C) Because the data was not standardized.
    *   [ ] D) Because the Gaussian kernel has infinite support (its tails never truly reach zero).

    **Answer: D) Because the Gaussian kernel has infinite support (its tails never truly reach zero).**

    **Explanation:**
    *   The mathematical formula for a Gaussian function extends from negative infinity to positive infinity. Therefore, even for a dataset of all positive numbers, the sum of the Gaussian kernels will result in a density curve that has a non-zero (though extremely small) value in the negative range.

49. **In the k-NN "find a good restaurant" analogy, asking the 50 closest people corresponds to:**
    *   [ ] A) A large k value with low bias (a more specific result).
    *   [ ] B) A large k value with high bias (a smoother, more general result).
    *   [ ] C) A small k value with high bias.
    *   [ ] D) A small k value with high variance.

    **Answer: B) A large k value with high bias (a smoother, more general result).**

    **Explanation:**
    *   Averaging over a large group (large k) smooths out the individual, noisy opinions and leads to a safe, popular, but potentially overly simple recommendation (high bias, low variance).

50. **Which statement is true about the relationship between k-NN and density?**
    *   [ ] A) k-NN is a method for making sparse data denser.
    *   [ ] B) k-NN has no relationship to the concept of data density.
    *   [ ] C) k-NN only works for data that is uniformly dense.
    *   [ ] D) k-NN classification implicitly assumes that classes with higher density near a point are more likely.

    **Answer: D) k-NN classification implicitly assumes that classes with higher density near a point are more likely.**

    **Explanation:**
    *   The logic of k-NN is based on local density. By taking a majority vote of the nearest neighbors, the algorithm is essentially betting that the new point belongs to the class that is most densely populated in its immediate vicinity.

    ### Back to Reading Content --> [Non-Parametric Techniques](../Non-ParametricTechnique.md)