<div style="text-align: justify;">

## Parameter Estimations

Parameter estimation is the process of using data from a sample to make an educated guess about a characteristic of a larger population. You're essentially using a small piece of the puzzle to figure out what the whole picture looks like.

These unknown population characteristics are called parameters. The numbers you calculate from your sample data to estimate these parameters are called statistics or estimates.

**An Intuitive Analogy: Tasting a Pot of Soup 🥣**

Imagine you're cooking a large pot of soup. You want to know if it's salty enough, but you can't taste the entire pot.

- The entire pot of soup is the population.
- The true saltiness of the entire pot is the parameter you want to know.
- You take a spoonful of soup to taste. This spoonful is your sample.
- The saltiness of your spoonful is your statistic or estimate.

Based on the taste of that one spoonful, you estimate the saltiness of the entire pot. If the spoonful is too bland, you infer the whole pot is bland and add more salt. You are using the sample to estimate the parameter.

The key idea is that we use a manageable sample to learn about a population that is too large or impractical to measure completely.

**Real-World Example: Estimating Average Student Height**

Let's say a university wants to know the average height of all its 30,000 students. Measuring every single student would be a huge task.

1. Define the Parameter: The parameter we want to estimate is the average height of all 30,000 students. This is a single, true value, but it's unknown.
2. Take a Sample: Instead of measuring everyone, the university randomly selects 100 students and measures their height. This group of 100 is the sample.
3. Calculate the Statistic: They calculate the average height of these 100 students. Let's say the result is 5 feet 8 inches. This is the sample mean, which is a statistic.
4. Estimate the Parameter: The university then uses this sample mean as an estimate for the population parameter. They conclude that the average height of all 30,000 students is approximately 5 feet 8 inches.

#### Two Main Types of Estimation

There are two primary ways to provide an estimate:

- Point Estimation

    This is when you give a single number as your best guess.
    Example: "Based on our sample, we estimate the average height of all students is 5 feet 8 inches."

    A point estimate is simple and direct, but it's very likely to be slightly off from the true population value. The chance that the average height of all 30,000 students is exactly 5 feet 8 inches is extremely small.
- Interval Estimation

    This is when you provide a range of values within which the true parameter is likely to fall, along with a level of confidence. This range is called a confidence interval.
    Example: "We are 95% confident that the true average height of all students is somewhere between 5 feet 7 inches and 5 feet 9 inches."

    An interval estimate is more informative and realistic. It acknowledges the uncertainty that comes from using a sample. It gives you a range of plausible values for the parameter and tells you how confident you are in that range.

In summary, parameter estimation is a fundamental concept in statistics where we use a small, manageable sample to make an informed guess about a characteristic of a much larger population.

### The Gaussian Case

"The Gaussian Case" in parameter estimation refers to the common and very useful scenario where we assume our sample data comes from a population that follows a Normal (or Gaussian) distribution.

A Gaussian distribution is the familiar "bell curve" shape and is completely described by just two parameters:

- The Mean (μ): This is the center of the bell curve, representing the average value of the population.
- The Variance (σ2): This measures the spread or width of the bell curve. A small variance means the data is tightly clustered around the mean, while a large variance means it's more spread out. The square root of the variance is the standard deviation (σ).

When we say we're in "the Gaussian case," our goal is to use a sample to make the best possible guesses for the true, unknown values of μ and σ2 for the entire population.

"Unknown μ" is the standard starting point for most real-world problems in statistics. It simply means that the true mean (average) of the entire population (μ) is not known and needs to be estimated. The entire process of parameter estimation we've been discussing is designed to solve this exact problem.

**The Problem: The Unknowable Average**

In virtually all practical scenarios, you cannot measure every single member of a population to find its true mean (μ).

- Manufacturing: A coffee company can't weigh every bag of coffee it ever produces to know the true average weight.
- Biology: A researcher can't measure the height of every single tree in a forest.
- Economics: An analyst can't survey every household to find the true average income.

In all these cases, the population mean (μ) is unknown.

**The Solution: Estimation from a Sample**

Since we can't know μ directly, we use a smaller, manageable sample to make an educated guess. This is done in two main ways:

- Point Estimation: Your Best Guess

    Your single best guess for the unknown population mean (μ) is the sample mean (xˉ). You calculate the average of your sample, and that becomes your estimate.

    Example: You take a sample of 30 coffee bags and find their average weight is 498 grams. This sample mean, xˉ=498g, is your point estimate for the unknown true mean, μ.

- Interval Estimation: Your Plausible Range

    You know your point estimate of 498g is probably not exactly correct. An interval estimate (or confidence interval) provides a range of values that likely contains the unknown μ.

    Example: Based on your sample, you calculate a 95% confidence interval of [496g, 500g]. This is a much more useful statement. It tells you that while your best guess is 498g, you are 95% confident that the true average weight of all coffee bags is somewhere between 496g and 500g. This range accounts for the uncertainty that comes from using a sample.

In short, having an "unknown μ" is the normal state of affairs. The techniques of calculating the sample mean and constructing a confidence interval are the statistical tools we use to make reliable inferences about this unknown value.

"Unknown μ and Σ" refers to the most common scenario in multivariate statistics, where you want to estimate the characteristics of a population that has multiple variables.

Here, you're trying to figure out two things from a sample:

- μ (Mu): The unknown mean vector of the entire population. This tells you the average value for each variable.
- Σ (Sigma): The unknown covariance matrix of the population. This tells you how each variable spreads out (its variance) and how it relates to or moves with the other variables (its covariance).

**The Problem: Understanding Multiple Features at Once**

Imagine you want to study the physical characteristics of all university students. You're interested in two variables: height and weight.

- The population mean vector (μ) is a pair of numbers: {true average height of all students, true average weight of all students}.
- The population covariance matrix (Σ) is a small grid of numbers that describes the spread of heights, the spread of weights, and the relationship between height and weight (e.g., as height increases, weight tends to increase).

Measuring every single student to find the true μ and Σ is impossible. They are unknown.

**The Solution: Estimation from a Sample**

You take a random sample of 100 students and record their height and weight. You then use this sample to estimate the unknown population parameters.

1. Estimating the Mean Vector (μ)

    Your best guess for the unknown population mean vector (μ) is the sample mean vector (xˉ).

    You simply calculate the average for each variable in your sample separately and put them into a vector.
    - Average height of the 100 students = 170 cm
    - Average weight of the 100 students = 65 kg

    So, your point estimate for μ is the vector:

    xˉ=[17065​]

    This is your best guess for the true average height and weight of the entire student population.

2. Estimating the Covariance Matrix (Σ)

    Your best guess for the unknown population covariance matrix (Σ) is the sample covariance matrix (S).

    This matrix captures the spread and relationships within your sample data. It's a bit more complex to calculate, but it would look something like this:

    $$
    S = \begin{bmatrix}
    \text{Variance of Height} & \text{Covariance of Height \& Weight} \\
    \text{Covariance of Height \& Weight} & \text{Variance of Weight}
    \end{bmatrix}
    $$

    - Variance (on the diagonal): These numbers tell you how spread out the height and weight values are, respectively. A larger number means more variation.
    - Covariance (on the off-diagonal): This number tells you how height and weight move together. A positive covariance would confirm our intuition that taller people tend to be heavier.

    Just like in the single-variable case, the formulas for calculating these sample values use a denominator of n-1 to ensure the estimate is unbiased.

In summary, when both μ and Σ are unknown, you are tackling a multivariate problem. You use the sample mean vector to estimate the population's central point and the sample covariance matrix to estimate the population's spread and the interconnections between its variables.


### Bias

In statistics, bias refers to the tendency of an estimator to consistently over or underestimate the true value of a population parameter. An estimator is called unbiased if, on average, its guesses hit the true value.

The relationship between bias and Maximum Likelihood Estimation (MLE) is that while MLE is a powerful method for finding estimates, it does not guarantee that the resulting estimator will be unbiased. Sometimes, the parameter that makes your observed data most likely is systematically a little bit off from the true population parameter.



### Quiz --> [Quiz](./Quiz/ParameterEstimationQuiz.md)

### Previous Topic --> [Logistic Regression](./LogisticRegression.md)
### Next Topic --> [PCA - Principal Component Analysis](./PrincipalComponentAnalyses.md)
</div>