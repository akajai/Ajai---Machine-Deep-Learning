<div style="text-align: justify;">

## Non-Parametric Techniques

### Density Estimation

Imagine you have a collection of data points, like the heights of all the students in a school. Density estimation is the process of taking this scattered data and creating a smooth curve or shape that shows where the data points are most concentrated. Think of it as creating a "population density map" for your data. Instead of showing where people live, it shows where your data "lives."

In simpler terms, it's a way of guessing the underlying pattern or distribution from which your data was drawn. It helps you see the "bigger picture" behind the raw numbers.

Think of it like creating a population density map from a list of addresses. Instead of seeing individual dots for each house, you get a shaded map showing the high-density urban centres and low-density rural areas. Density estimation does the same for your data.

### Parametric Distribution

A parametric distribution is a statistical distribution whose shape and characteristics are completely defined by a fixed, finite number of parameters.

In simpler terms, if you know the values of these few key parameters (like the mean and standard deviation), you know everything there is to know about the distribution.

Think of it like a recipe. 🧑‍🍳 A "Normal Distribution" is a recipe for a bell curve. The parameters are the ingredients and their amounts:

- Mean (3μ): Tells you where to center the curve on a number line.
- Standard Deviation (5σ): Tells you how wide or narrow the curve should be.

Once you set these two parameters, the exact shape and position of the bell curve are locked in.7 You don't need any more information.

**Key Characteristics**

- Assumptions are Made: To use a parametric model, you must assume that your data follows a specific shape or distribution (e.g., you assume it's bell-shaped, U-shaped, etc.).8
- Efficiency: They are mathematically simpler and require less data to estimate the distribution accurately compared to non-parametric methods.
- Defined by Parameters: The entire distribution can be summarized by a small set of numbers (the parameters).9

**Common Examples of Parametric Distributions**

Each of these is a "family" of distributions, and the parameters define a specific member of that family.

- Normal Distribution (Gaussian): Defined by its mean (10μ) and standard deviation (11σ).12 It's the classic symmetric "bell curve" used for things like heights, blood pressure, and measurement errors.13
- Binomial Distribution: Defined by the number of trials (n) and the probability of success (p).14 It's used for outcomes that are binary (e.g., success/failure, heads/tails) across multiple trials.
- Poisson Distribution: Defined by a single parameter, the average rate (15λ).16 It's used for counting the number of events happening in a fixed interval of time or space (e.g., the number of customers arriving at a store in an hour).

### Parametric Density Estimation

Parametric density estimation is a method where you first assume your data follows a specific, known probability distribution (like a Normal, Binomial, or Poisson distribution) and then estimate the parameters of that distribution from your data.

Think of it like using a cookie cutter. 🍪

- The Cookie Cutter: This is your chosen distribution family (e.g., a bell-shaped cutter for a Normal distribution). You're assuming your data's true shape is something like this.
- The Dough: This is your raw dataset.
- The Process: Your job is to find the best way to place and size the cutter to fit the dough. Estimating the parameters (like the mean and standard deviation) is like finding the perfect centre and width for your cutter.

Once you have the parameters, you have a complete, simple mathematical formula that describes your data's density.

Let's use an example of estimating the distribution of heights for a group of adult men.

1. Step 1: Choose a Distribution Model

    First, you make an educated guess about the shape of your data. You might plot a quick histogram and see that it looks roughly symmetric and bell-shaped. Based on this, you assume the data follows a Normal distribution. This is the most critical step.
2. Step 2: Estimate the Parameters from the Data

    Now that you've chosen the Normal distribution "family," you need to find the specific parameters that fit your data best. For a Normal distribution, the two parameters are:

    - Mean (μ): The centre of the distribution.
    - Standard Deviation (σ): The spread or width of the distribution.

    You simply calculate the average height and the standard deviation from your dataset. Let's say you find the mean is 177 cm and the standard deviation is 8 cm.

3. Step 3: Get the Final Density Function

    You've done it! Your parametric density estimate is the mathematical formula for a Normal distribution with μ=177 and σ=8. You can now use this compact model to answer questions like, "What's the probability of finding a man who is taller than 193 cm?"

### Non-Parametric Estimation

Non-parametric estimation is a method of understanding the shape of your data without making any prior assumptions about that shape. Instead of trying to fit the data into a predefined box (like a bell curve), this approach lets the data speak for itself, building a custom shape directly from the data points.

Think of it as creating a portrait. 🎨

- Parametric estimation is like a "paint-by-numbers" kit for a portrait. You assume the final picture will fit the pre-drawn lines, and your only job is to fill in the specific colours (the parameters).
- Non-parametric estimation is like being a sketch artist. You start with a blank canvas and carefully draw the contours and features exactly as you see them, letting the subject's (the data's) unique structure dictate the final image.

The core idea is flexibility. The model's complexity grows and adapts based on the amount of data you have.

### Histogram Based Density Estimation

A histogram-based density estimation is a simple method of visualizing the probability distribution of a dataset by sorting the data into a series of intervals, or "bins," and counting how many data points fall into each one. It's a non-parametric technique because it doesn't assume any underlying shape for the data.

Think of it as sorting a collection of coins into different jars based on their value. Each jar is a bin, and the number of coins in each jar gives you a sense of the distribution of coin values. 🪙

Think of it as sorting a collection of coins into different jars based on their value. Each jar is a bin, and the number of coins in each jar gives you a sense of the distribution of coin values. 🪙

Let's use a simple example: the exam scores of 20 students.

Data: [85, 92, 88, 72, 75, 95, 100, 68, 79, 81, 90, 76, 83, 71, 98, 86, 77, 89, 93, 73]

1. Step 1: Choose the Bin Width

    This is the most crucial step. The bin width determines the size of each interval. Let's choose a bin width of 10 points. This gives us bins like 60-70, 70-80, 80-90, and so on.

2. Step 2: Create the Bins and Count the Frequencies

    Now, we go through our data and count how many scores fall into each 10-point range.

    - 60-70: 68 (1 student)
    - 70-80: 72, 75, 79, 76, 71, 77, 73 (7 students)
    - 80-90: 85, 88, 81, 83, 86, 89 (6 students)
    - 90-100: 92, 95, 90, 98, 93 (5 students)
    - 100-110: 100 (1 student)

3. Step 3: Normalize to Get the Density (Optional but Important)

    To turn the frequency count into a probability density, the height of each bar is normalized. The rule is that the total area of all bars must sum to 1.

    The height of each bar is calculated as:

    $$
    \text{Density} = \frac{\text{Frequency}}{\text{Total Number of Samples} \times \text{Bin Width}}
    $$

    Let's calculate the density for the 70-80 bin:

    $$
    \text{Density (70-80)} = \frac{7}{20 \times 10} = \frac{7}{200} = 0.035
    $$

    The choice of bin width can drastically change the look and interpretation of the histogram.

    - Too Wide: If we chose a bin width of 20 (e.g., 60-80, 80-100), we would lose a lot of detail. We might see one big lump and miss the finer structure of the score distribution.
    - Too Narrow: If we chose a bin width of 2 (e.g., 70-72, 72-74), the histogram would look very noisy and jagged, making it hard to see the overall trend.


### Bin Width

Selecting the right bin width is crucial for creating a useful histogram, as it directly controls how "smooth" the resulting density estimate appears. There's no single perfect answer, but the choice is guided by several formal rules and a core statistical concept.

The bin width in a histogram serves the exact same purpose as the smoothing parameter (often called bandwidth).

Choosing a bin width is a classic example of the bias-variance trade-off.

- Wide Bins (High Bias, Low Variance): If your bins are too wide, you lump too much data together. This creates a very smooth, simple-looking histogram that hides important details (like multiple peaks). It's biased because it oversimplifies the true distribution. It has low variance because changing a few data points won't change the look of the histogram much.
    - Analogy: Looking at a city from a satellite 🛰️. You see the overall shape but miss all the streets and buildings.
- Narrow Bins (Low Bias, High Variance): If your bins are too narrow, the histogram becomes noisy and jagged. Every little fluctuation in the data creates a new peak or valley, making it hard to see the underlying trend. It has low bias because it follows the data closely, but high variance because just a small change in the data can drastically alter the shape.
    - Analogy: Looking at the city with a microscope 🔬. You see every crack in the pavement but have no sense of the city's layout.

The goal is to find a bin width that is the "Goldilocks" choice—not too wide, not too narrow—that minimizes both errors and best reveals the data's true shape.

### Kernal Density Estimation

Kernel Density Estimation (KDE) is a non-parametric method for visualizing the underlying distribution of a dataset. In simple terms, it's a way to create a smooth, continuous curve from a set of scattered data points, showing where the data is most concentrated.

Think of it as creating a "smooth histogram." Instead of putting data into discrete bins, KDE builds a flowing, nuanced shape that lets the data speak for itself.

This is the most intuitive way to understand KDE.

1. The Pond is Your Graph: Imagine a perfectly still pond, which represents your blank graph or number line.
2. Each Data Point is a Pebble: You take each data point from your dataset and drop a small, identical pebble into the pond at that exact location.
3. Each Pebble Creates a Ripple: Every pebble creates a small, smooth ripple (a "bump") around it. This ripple is the kernel.
4. Ripples Add Up: As you drop all your pebbles, their ripples start to overlap. In areas where you dropped many pebbles close together, the individual ripples combine to form large, high waves. In areas with few pebbles, the water remains calm.
5. The Final Water Surface is the KDE: The final, wavy shape of the water's surface is your kernel density estimate. The peaks of the waves show where your data points are most clustered.

**Example: Bus Arrival Times 🚌**

Imagine you've recorded the arrival times of a bus at a specific stop for a month. The bus is scheduled to arrive at 8:00 AM. Your data points are minutes before or after 8:00 (e.g., -2, 1, 0, 5, -1, 3, ...).

- A Histogram might show you a large bar between 0 and 5 minutes, but it would be blocky.
- A Kernel Density Estimate would create a smooth curve from this data. It would likely show a high peak around the 1-minute mark, indicating the bus is most frequently a minute late. The curve would also show you the shape of the delays—perhaps a "long tail" in the positive direction, showing that while the bus is rarely early, it's sometimes very late. This smooth shape gives a much more detailed and intuitive understanding of the bus's punctuality than a simple histogram.

### kernel function

A kernel function is a mathematical shortcut that calculates the similarity between two data points in a higher-dimensional space, without ever actually having to transform the data into that space.

In the simplest terms, it's a "similarity calculator." You give it two data points, and it returns a number that tells you how alike they are. The clever part is that it does this by implicitly viewing the points from a more complex perspective or a higher dimension.

Imagine you're judging a fruit contest. You have apples and oranges, but they are all mixed up on a table.

- The Problem (1D): If you try to separate them based on a single feature, like their weight, you'll fail. An apple and an orange can have the same weight, so you can't just draw a single line to separate them.
- The Kernel Function's Magic: Now, instead of just weighing them, you use a "fruit similarity" function (our kernel). This function might implicitly consider multiple features at once, like weight, color, and texture.

It answers the question: "How similar are Fruit A and Fruit B based on this complex set of features?"

By using this kernel, you are effectively looking at the fruit in a higher-dimensional space (a "feature space" that includes weight, color, and texture). In this new space, the apples and oranges form distinct, easily separable clusters. The kernel function gets you this result without you having to manually create the new "weight-color-texture" graph. This shortcut is known as the kernel trick.

### Parzen Window

A Parzen window, also known as the Parzen-Rosenblatt window method, is a non-parametric technique used to estimate a probability density function from a set of data points. It's essentially the same concept as Kernel Density Estimation (KDE).

Think of it as a sophisticated way to build a smooth curve representing your data's distribution by placing a small "window" or "bump" on top of each data point and then summing all the bumps together.

The Parzen window method builds a density function, f(x), by adding up several kernel functions centered at each data point, xi​. The final estimate is the average of these kernels.

### Box Kernal

A box kernel, also known as a rectangular or uniform kernel, is a simple function used in kernel density estimation that gives uniform, or equal, weight to all data points within a specific distance (the bandwidth) and zero weight to all points outside that distance.

Think of it as an "all-or-nothing" window. If a data point is inside the box, it's counted fully. If it's outside, it's completely ignored.

The box kernel is defined by a simple rule. For a data point x being evaluated, the kernel gives a constant value (usually 1/2) to any neighboring data point xi that is within a certain bandwidth h, and a value of 0 to any neighbor that is further away.

Its shape is a simple rectangle or box, hence the name. It is flat on top and drops vertically to zero at the edges of the window.

Imagine you're standing on a street and want to get a sense of the opinions of the people immediately around you.

- Your Position: This is the point x where you are estimating the density.
- The Box Kernel: You decide you'll only talk to people within a 10-foot radius of you. This 10-foot circle is your "box."
- The Bandwidth (h): The radius of the circle (10 feet) is your bandwidth.
- The Process: You give every single person inside your 10-foot circle equal importance. You don't care if someone is 1 foot away or 9 feet away; their opinion is counted just the same. Anyone standing 11 feet away is completely ignored, as if they don't exist.

The final "density" of opinions is just the average of the opinions of the people inside your circle. This is how the box kernel works—it creates a density estimate that is essentially a moving average. The resulting density curve is often blocky and has sharp steps, unlike the smooth curves produced by more popular kernels like the Gaussian kernel.

### K in terms of Kernel Function

In the context of kernel functions, K is the standard mathematical symbol used to represent the kernel function itself. It's a placeholder for a specific kernel, much like f is a common symbol for a function in algebra.

Think of K as a "similarity calculator." It's a function that takes two data points (let's call them x₁ and x₂) as input and outputs a single number that quantifies how similar they are.

So, when you see the notation **K(x₁, x₂) **, it means:

"Calculate the similarity between data point x₁ and data point x₂ using the kernel function K."

### The Probability Density in terms of Kernel

The probability density at a specific point, x, is estimated by summing the contributions of all nearby data points, where each data point's contribution is determined by a kernel function.

In simpler terms, you build the probability density curve by placing a smooth "bump" (the kernel) on top of every data point and then adding up all the bumps.

$$
\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K \left( \frac{x - x_i}{h} \right)
$$

- f^​(x): This is what you're trying to find—the estimated probability density at a specific point, x.
- n: The total number of data points you have.
- h: The bandwidth (or smoothing parameter), which controls the width of the bumps.
- xi​: Represents each individual data point in your dataset.
- K: The kernel function. This is the core component that defines the shape of the "bump" placed on each data point. It's a function that takes the distance between x and a data point xi and returns a value.

### Gaussian Kernel

A Gaussian kernel is a function used to weigh the influence of data points based on their distance from a central point. Its defining characteristic is that it assigns the highest weight to the central point, with the weight diminishing smoothly and symmetrically in a "bell curve" shape as the distance increases.

In simple terms, it's a "similarity calculator" that assumes points closer to each other are much more similar than points that are far apart.

The Gaussian kernel is named after the Gaussian (or normal) distribution, as it uses the same mathematical form. The formula for a simple 1D Gaussian kernel is:

$$
K(x, x_i) = \frac{1}{\sqrt{2\pi\sigma}} e^{-\frac{(x - x_i)^2}{2\sigma^2}}
$$

- x: The point where you are calculating the density or influence.
- xi​: The center of the kernel (the location of a data point).
- σ (sigma): The standard deviation, which acts as the bandwidth or smoothing parameter. This is the most critical part.
    - A small σ results in a narrow, steep bell curve, meaning only very close points have a significant influence.
    - A large σ results in a wide, gentle bell curve, meaning points that are farther away still have a considerable influence.

The key part of the function is the exponent, which calculates the squared distance between the points. The further apart they are, the more negative the exponent becomes, causing the function's value to drop rapidly towards zero.

Think of a streetlight on a foggy night.

- The Streetlight (xi​): This is your data point.
- The Light's Glow: This is the Gaussian kernel. The light is brightest right under the lamp and fades away smoothly and symmetrically in all directions.
- Your Position (x): You are standing somewhere on the street.
- The Brightness You See: The amount of light you experience is the kernel's output. If you stand right under the lamp, the brightness is at its maximum. As you walk away, the brightness decreases in a bell-curve fashion.
- The Bandwidth (σ): This is like the power of the streetlight's bulb. A high-powered bulb (large σ) will cast a wide, gentle glow that illuminates a large area. A low-powered bulb (small σ) will cast a sharp, concentrated light that fades very quickly.


### Triangular Kernel

The Triangular kernel assigns weights that decrease linearly from a maximum at the center to zero at the edges of the window.

- Shape: A triangle.
- Key Feature: It's a step up from the Box kernel, giving more importance to closer points, which results in a continuous (but not perfectly smooth) density estimate.

The main benefit of Kernel Density Estimation (KDE) is its ability to create a smooth, detailed visualization of a data distribution without prior assumptions, while its main drawback is its sensitivity to the choice of the smoothing parameter (bandwidth) and its higher computational cost for large datasets.

**✅ Pros (Advantages)**

- Smooth and Continuous: KDE produces a smooth, continuous curve, which is often a more natural and interpretable representation of the underlying data distribution than a blocky histogram.
- No Assumptions Required (Non-parametric): It does not assume that the data follows a specific distribution (like a Normal distribution). This flexibility allows it to accurately model complex distributions with multiple peaks, skewness, or other unusual features.
- Reveals Detailed Structure: Because it's not constrained by bins, KDE can reveal finer details in the data, such as small peaks or bumps that a histogram might miss.
- Mathematically Convenient: The resulting density function is mathematically well-defined, making it useful for further statistical analysis.

**❌ Cons (Disadvantages)**

- Sensitive to Bandwidth Choice: This is the most significant limitation. The entire shape of the density estimate is critically dependent on the choice of the bandwidth (h). An inappropriate bandwidth can lead to a curve that is either too noisy (under-smoothed) or too general (over-smoothed), resulting in a misleading interpretation.
- Computationally Intensive: Calculating the density estimate requires summing up kernel functions for every data point, which can be slow and memory-intensive for very large datasets compared to a simple histogram.
- Can Be Difficult to Interpret: The "curse of dimensionality" makes KDE difficult to implement and interpret for data with many features (dimensions). The concept of density becomes less meaningful in high-dimensional spaces.
- Can Extend to Impossible Values: A standard Gaussian kernel has infinite support, meaning the resulting density curve can extend to values that are not logically possible (e.g., estimating a small, non-zero density for a negative house price).


### Nearest Neighbour Method

The Nearest Neighbour method is a non-parametric technique for estimating the probability density at a specific point by examining the distance to its closest data points.

Unlike the Parzen window (KDE) method, which uses a fixed bandwidth h, the Nearest Neighbor method uses a fixed number of points, k, and lets the volume containing them vary.

The core idea is that the density in a region is high if the data points are tightly packed, and low if they are spread far apart. This method quantifies that intuition.

To estimate the density at a point x:

1. Choose k: First, you decide on a value for k, which is the number of neighbors you'll consider (e.g., k=5).
2. Find the k-th Nearest Neighbor: From your target point x, you find the distance to its k-th nearest data point. Let's call this distance d_k.
3. Define the Volume: This distance d_k becomes the radius of a sphere (or a hyper-sphere in more than 3 dimensions) centered at x. The volume V of this sphere is calculated.
4. Calculate the Density: The density at x is then estimated by the simple formula: f^​(x)=nVk​
    
    Where:
    - k is the number of neighbors.
    - n is the total number of data points.
    - V is the volume of the sphere containing the k neighbors.

Imagine you are a drone flying over a park, and you want to create a density map.

- Your Position (x): The point on the map where you want to estimate the crowd density.
- k: You decide to always base your estimate on the 5 closest people to your target point.
- The Process:
    - You hover over a point in a crowded picnic area. The 5th closest person is only a few feet away. This defines a small circle (your volume V). Because V is tiny, the density calculation (k/nV) gives a very high value.
    - Next, you move to an open field with only a few people. To find the 5th closest person, you have to search a much wider area. This defines a huge circle. Because V is large, the density calculation gives a very low value.

The result is a density estimate that adapts to the data: it uses a narrow window in dense regions and a wide window in sparse regions.

**The Role of k (The Smoothing Parameter)**

Just like the bandwidth h in KDE, k is the smoothing parameter in the Nearest Neighbor method.

- Small k: This results in a very noisy, detailed estimate because the volume V can change dramatically from one point to the next. (High variance, low bias).
- Large k: This results in a very smooth, less detailed estimate because the volume V will be large and average over a wider area. (Low variance, high bias).

**Finding  the labels**

n the context of the k-Nearest Neighbors (k-NN) algorithm, "finding the labels" refers to the final step where the algorithm assigns a predicted label or value to a new data point based on the labels of its k closest neighbors.

This process differs depending on whether you are doing classification or regression.

**k=1 Nearest Neighbor rule**

The k=1 Nearest Neighbor rule is the simplest version of the k-NN algorithm where a new data point is assigned the exact same class as its single closest neighbor from the training data.

1. For a new, unclassified data point, calculate its distance to every point in the training dataset.
2. Identify the single training data point that is closest (the "nearest neighbor").
3. Assign the class label of that single nearest neighbor to the new data point.

This method creates a decision boundary that is essentially a mosaic of cells (called Voronoi cells) around each training data point. Any new point that falls into a specific data point's "cell" gets classified with that point's label.

**K control the smoothing**

In the k-Nearest Neighbors algorithm, k is the primary parameter that controls the amount of smoothing, which in turn determines the complexity of the model.

Think of k as a tuning knob for your model's sensitivity. 🎛️

The value of k determines how many neighbors get to "vote" on the classification of a new point. This directly influences the shape of the decision boundary.

- Small k (e.g., k=1)
    
    - Effect: Low smoothing. The decision is based on just one or a few neighbors.
    - Result: The model is highly sensitive to noise and individual data points. This leads to a very complex and jagged decision boundary that closely follows the training data.
    - Risk: Overfitting. The model learns the noise in the data, not just the underlying trend, so it doesn't generalize well to new data.
- Large k (e.g., k=25)

    - Effect: High smoothing. The decision is averaged over a large number of neighbors.
    - Result: The model is less sensitive to individual data points and noise. This leads to a much smoother, more generalized decision boundary.
    - Risk: Underfitting. If k is too large, the model might over-simplify the boundary between classes, missing the local structure and patterns in the data.


Imagine you're in a new city and want to find a good restaurant.

- Small k (k=1): You ask the single closest person. You might get a great, unique recommendation, or you might have found the one person with terrible taste. Your decision is highly variable.
- Large k (k=50): You poll the 50 closest people. The final recommendation will be a very safe, popular choice (like a well-known chain), averaging out any unique or terrible opinions. Your decision is much smoother and less risky.

This choice of k is a classic example of the bias-variance trade-off:

- Small k: Low bias (it follows the data closely) but high variance (it's unstable and sensitive to noise).
- Large k: High bias (it makes simplifying assumptions) but low variance (it's stable and consistent).

Finding the optimal k means finding the right balance between these two extremes for your specific dataset.

**U-Shaped Dataset**

A U-shaped dataset describes a bimodal distribution where the highest frequencies of data points are at the extremes of the range, and the lowest frequency is in the middle.

When plotted as a histogram or density curve, the shape resembles the letter "U".

- Two Peaks: The distribution has two distinct peaks (modes), but unlike a typical bimodal distribution where the peaks are somewhere along the range, in a U-shaped distribution, the peaks are at the very ends of the data range.
- Low Centre: The values in the middle of the range are the least common.
- Polarization: This shape often indicates a polarization or division within the data. It suggests that most of the data points cluster into two distinct, opposing groups.

**Real-World Examples**

- Customer Satisfaction Scores: On a 1-to-5 star rating system, you often see many 5-star ratings (from happy customers) and many 1-star ratings (from unhappy customers), with fewer ratings in the 2-4 star range.
- Political Opinions: When surveying opinions on a controversial topic, you might find many people strongly agree or strongly disagree, with few holding a neutral position.
- Cloud Cover: The percentage of cloud cover is often either close to 0% (clear sky) or close to 100% (overcast), with partly cloudy days being less frequent.

A U-shaped distribution can be misleading if you only look at simple summary statistics. For example, the mean or median of a U-shaped dataset will fall right in the centre of the range where the data is scarcest, making it a very poor representation of a "typical" value.




### Quiz --> [Non-Parametric Techniques Quiz](./Quiz/Non-ParametricTechniqueQuiz.md)

### Previous Topic --> [PCA - Principal Component Analyses](./PrincipalComponentAnalyses.md)
### Next Topic --> [Decison Tree](./DecisionTree.md)
</div>