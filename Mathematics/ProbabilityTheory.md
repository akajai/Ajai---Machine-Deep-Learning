<div style="text-align: justify;">

## 🎲 The Basics of Probability

Probability is simply a way to measure how likely something is to happen. We give it a number between 0 and 1.

* **0** means the event is **impossible** (e.g., the probability of rolling a 7 on a standard six-sided die).
* **1** means the event is **certain** (e.g., the probability that the sun will rise tomorrow).
* **0.5** means it's a 50/50 chance (e.g., the probability of getting "heads" when flipping a fair coin).

### Key Terms

* **Experiment:** An action with an uncertain result, like flipping a coin or rolling a die.
* **Outcome:** A single possible result of an experiment, like getting "Tails".
* **Sample Space:** The complete set of all possible outcomes. For a die, it's {1, 2, 3, 4, 5, 6}.
* **Event:** A specific outcome or a collection of outcomes you're interested in, like "rolling an even number" {2, 4, 6}.

The basic formula for calculating the probability of an event (E) is:

$$P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

**Example: Rolling a Die** 
What's the probability of rolling a number greater than 4?
* Favorable outcomes: {5, 6} (That's 2 outcomes).
* Total outcomes: {1, 2, 3, 4, 5, 6} (That's 6 outcomes).
* $P(\text{rolling} > 4) = \frac{2}{6} = \frac{1}{3} \approx 0.33$

### 🔗 Related Events: Conditional, Joint, and Dependent Probability

### Dependent vs. Independent Events

* **Independent Events:** The outcome of one event doesn't affect the outcome of another. Think of flipping a coin twice. The first flip has no impact on the second.
* **Dependent Events:** The outcome of the first event *changes* the probability of the second event.

**Real-Life Example: Drawing Cards**
Imagine a standard 52-card deck.
* **Event A:** Drawing a King first. The probability is $P(A) = \frac{4}{52}$.
* **Event B:** Drawing a second King *without* putting the first one back.

Since you didn't replace the first card, there are now only 51 cards left, and only 3 of them are Kings. The probability of Event B *depends* on Event A happening. This is a classic case of **dependent events**.

### Conditional Probability: Probability Under a Condition

Conditional probability answers the question, "What is the probability of A happening, *given that* B has already happened?". It's written as $P(A|B)$.

The formula is:

$$P(A|B) = \frac{P(A \text{ and } B)}{P(B)}$$

**Example: The Bag of Marbles** 
You have a bag with 3 Red and 2 Blue marbles. Let's find the probability of drawing a Red marble second, *given that* you drew a Red marble first (and didn't replace it).

* **Event B (the condition):** Drawing a Red marble first. $P(\text{Red first}) = \frac{3}{5}$.
* After this, the bag has 2 Red and 2 Blue marbles (4 total).
* **Event A (what we're measuring):** Drawing a Red marble second.
* The probability of this is now $P(\text{Red second} | \text{Red first}) = \frac{2}{4} = \frac{1}{2}$.

The probability changed from $\frac{3}{5}$ to $\frac{1}{2}$ because the events were dependent. If the events were independent, $P(A|B)$ would just be equal to $P(A)$.

### Joint Probability: The Chance of Two Things Both Happening

Joint probability is the chance of two or more events happening at the same time or in sequence. It's written as $P(A \text{ and } B)$ or $P(A \cap B)$.

The formula depends on whether the events are independent or dependent.

* **For Independent Events**:
    $P(A \text{ and } B) = P(A) \times P(B)$ 
    * **Example:** The probability of flipping "Heads" ($P=0.5$) AND rolling a "6" ($P=1/6$) is $0.5 \times \frac{1}{6} = \frac{1}{12}$.

* **For Dependent Events**:
    $P(A \text{ and } B) = P(A) \times P(B|A)$ 
    * **Example:** The probability of drawing a King AND then a second King from a deck is $\frac{4}{52} \times \frac{3}{51} \approx 0.0045$.

### 🧠 Bayes' Rule: Updating Your Beliefs with Evidence

Bayes' Rule is a formula that lets us update our initial beliefs about something based on new evidence. It connects conditional probabilities, showing how $P(A|B)$ is related to $P(B|A)$.

**The Formula**:
$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Let's break it down:
* $P(A|B)$ **(Posterior):** What we want to find. The probability of our hypothesis (A) being true, *given* the new evidence (B).
* $P(B|A)$ **(Likelihood):** The probability of seeing the evidence (B) if our hypothesis (A) is true.
* $P(A)$ **(Prior):** Our initial belief in the hypothesis (A) *before* seeing any evidence.
* $P(B)$ **(Evidence):** The total probability of observing the evidence (B) under all circumstances.

**Real-Life Example: The Rare Disease Test** 
Imagine a medical test for a disease that only 1 in 1000 people have.
* **Prior belief:** $P(\text{Disease}) = 0.001$. Your initial chance is very low.
* The test is 99% accurate if you have the disease: $P(\text{Positive Test} | \text{Disease}) = 0.99$.
* The test has a 5% "false positive" rate: $P(\text{Positive Test} | \text{No Disease}) = 0.05$.

You take the test and it comes back **POSITIVE**. What's the real chance you have the disease? We want to find $P(\text{Disease} | \text{Positive Test})$.

1.  **Likelihood:** $P(\text{Positive Test} | \text{Disease}) = 0.99$.
2.  **Prior:** $P(\text{Disease}) = 0.001$.
3.  **Evidence:** We need to find the total probability of anyone getting a positive test. This can happen in two ways: you have the disease and test positive, OR you don't have it and get a false positive.
    * $P(\text{Positive Test}) = [P(\text{Positive}| \text{Disease}) \times P(\text{Disease})] + [P(\text{Positive}| \text{No Disease}) \times P(\text{No Disease})]$ 
    * $P(\text{Positive Test}) = (0.99 \times 0.001) + (0.05 \times 0.999) = 0.00099 + 0.04995 = 0.05094$.

4.  **Apply Bayes' Rule:**
    $$
    P(\text{Disease} | \text{Positive Test}) = \frac{0.99 \times 0.001}{0.05094} \approx 0.0194
    $$
    Even with a positive test, your chance of having the disease is only about **1.94%**!  This is because the disease is so rare that false positives, while individually uncommon, make up most of the positive results.


### 🔢 Random Variables: Turning Outcomes into Numbers

A random variable is a variable whose value is a numerical outcome of a random phenomenon. We use a capital letter (like X) to denote the variable itself, and a lowercase letter (x) for its specific values.

There are two main types: Discrete and Continuous.

### 1. Discrete Random Variables

These can only take on a finite or countable number of distinct values (usually integers). Think of things you can **count**.
* **Examples:** The number of heads in 3 coin flips (can be 0, 1, 2, or 3), the number of defective items in a batch, the number on a rolled die.

#### Probability Mass Function (PMF)

The PMF gives the probability that a discrete random variable is exactly equal to some value. It's a list or formula that maps each possible outcome to its exact probability.

**Example: Two Coin Flips** 
Let X be the number of heads in two flips.
* Possible outcomes: HH, HT, TH, TT.
* Possible values for X: {0, 1, 2}.
* The PMF, p(x), would be:
    * $P(X=0) = P(TT) = 1/4$ 
    * $P(X=1) = P(HT \text{ or } TH) = 2/4 = 1/2$ 
    * $P(X=2) = P(HH) = 1/4$ 

A valid PMF must satisfy two rules:
1.  All probabilities must be between 0 and 1 ($0 \le p(x) \le 1$).
2.  The sum of all probabilities must equal 1 ($\sum p(x) = 1$).

### 2. Continuous Random Variables

These can take on an infinite number of possible values within a given range. Think of things you can **measure**.
* **Examples:** A person's height, the temperature of a room, the time it takes to run a race.

A key concept: the probability that a continuous random variable is *exactly* equal to a single value is **zero**. Why? Because there are infinite possibilities. The chance of someone being *exactly* 180.0000... cm tall is effectively zero. Instead, we talk about the probability of the value falling within a *range*.

#### Probability Density Function (PDF)

For continuous variables, we use a PDF, denoted $f(x)$. The PDF itself is not a probability. Instead, the **area under the curve** of the PDF between two points gives you the probability that the variable will fall in that range.

$$P(a < X < b) = \int_{a}^{b} f(x) dx$$

A valid PDF must satisfy two rules:
1.  The function must always be non-negative ($f(x) \ge 0$).
2.  The total area under the entire curve must equal 1.

### Cumulative Distribution Function (CDF)

The CDF, $F(x)$, gives the accumulated probability of everything up to a certain point. It answers the question, "What's the probability that my random variable X will be less than or equal to a value x?"

$$F(x) = P(X \le x)$$

For a discrete variable, it's a "step function" that jumps up at each possible value. For a continuous variable, it's a smooth, non-decreasing curve that goes from 0 to 1.

### 📊 Common Probability Distributions

### Discrete Distributions

#### Bernoulli Distribution
The simplest one. It models a single trial with only two outcomes: success or failure.
* **Parameter:** $p$ (the probability of success).
* **Example:** A single free throw shot. Let X=1 for a success (making the shot) and X=0 for a failure (missing). If the player has a 75% success rate, then $p=0.75$.
* **PMF:** $P(X=x) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$.

#### Binomial Distribution
Models the number of successes in a *fixed number of independent Bernoulli trials*.
* **Parameters:** $n$ (number of trials) and $p$ (probability of success per trial).
* **Conditions:** Fixed trials, two outcomes, independent trials, constant probability of success.
* **Example:** A factory finds 10% of its light bulbs are defective ($p=0.10$). You test a batch of 5 bulbs ($n=5$). What is the probability that *exactly one* is defective?
* **PMF:** $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$ 
    * $\binom{n}{k}$ is the number of ways to choose $k$ successes from $n$ trials.
    * $P(X=1) = \binom{5}{1} (0.10)^1 (0.90)^4 = 5 \times 0.10 \times 0.6561 = 0.32805$.
    * There's a **32.8%** chance that exactly one bulb is defective.

#### Poisson Distribution
Models the number of events occurring in a fixed interval of time or space, given a known average rate.
* **Parameter:** $\lambda$ (lambda), the average number of events per interval.
* **Example:** A call center receives an average of 5 calls per minute ($\lambda=5$). The Poisson distribution can tell you the probability of receiving exactly 0, 1, 2, 3, etc., calls in the next minute.

#### Multinomial Distribution
A generalization of the Binomial distribution. It models an experiment with a fixed number of trials, but where each trial has *more than two* possible outcomes.
* **Parameters:** $n$ (number of trials) and a vector of probabilities $p_1, p_2, ..., p_k$ for each of the $k$ categories.
* **Example:** A restaurant finds customers choose a Burger 50% of the time, Pizza 30%, and Salad 20%. Out of the next 10 customers ($n=10$), what is the probability that 5 choose a Burger ($x_1=5$), 3 choose Pizza ($x_2=3$), and 2 choose Salad ($x_3=2$)?
* **PMF:** $P(X_1=x_1, ..., X_k=x_k) = \frac{n!}{x_1! x_2! ... x_k!} p_1^{x_1} p_2^{x_2} ... p_k^{x_k}$ 
    * For the example, the probability is about **8.51%**.

### Continuous Distributions

#### Uniform Distribution
The simplest continuous distribution. All outcomes within a certain range are equally likely. The PDF is just a flat line.
* **Parameters:** $a$ (min value) and $b$ (max value).
* **Example:** A random number generator that produces a number between 0 and 1. Any number has an equal chance of being generated.

#### Exponential Distribution
Models the time you have to wait *until* an event occurs, assuming events happen at a constant average rate. It is closely related to the Poisson distribution.
* **Parameter:** $\lambda$ (lambda), the rate parameter (e.g., events per hour).
* **Example:** If a hospital's emergency room gets an average of 10 patients per hour, the exponential distribution can model the time between each new patient's arrival.

#### Normal Distribution (The Bell Curve)
The most famous distribution in statistics. It's a symmetric, bell-shaped curve where most data clusters around the average. Many natural phenomena follow this pattern, like height, weight, and IQ scores.
* **Parameters:**
    * **Mean ($\mu$):** The center of the curve.
    * **Standard Deviation ($\sigma$):** How spread out the data is. A small $\sigma$ gives a tall, narrow curve; a large $\sigma$ gives a short, wide curve.
* **PDF Formula:**
    $$
    f(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
    $$
    * The $(x-\mu)/\sigma$ part is the **Z-score**, which measures how many standard deviations a point $x$ is from the mean. This is the core engine of the shape.
    * The $e^{-(\dots)^2}$ part creates the bell shape, peaking at the mean and tapering off symmetrically.
    * The $\frac{1}{\sigma\sqrt{2\pi}}$ part is a scaling factor that ensures the total area under the curve is exactly 1.

### 📈 Key Concepts in Statistics

### Expectation (Mean) and Variance

* **Expectation (E[X] or $\mu$):** The long-run average value of a random variable. It's a weighted average, where each outcome is weighted by its probability.
    * **For Discrete Variables:** $E[X] = \sum x \cdot P(X=x)$. You multiply each value by its probability and sum them all up. For a fair six-sided die, the expected value is 3.5.
    * **For Continuous Variables:** $E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$.

* **Variance ($\text{Var}(X)$ or $\sigma^2$):** Measures how spread out the data is from the mean.
    * **Low Variance:** Data points are clustered tightly around the mean.
    * **High Variance:** Data points are widely scattered.

### The Central Limit Theorem (CLT)

This is one of the most important theorems in statistics. It states that if you take large enough random samples from *any* population (even a non-normal one), the distribution of the **sample means** will be approximately normal. This is why the normal distribution is so common—it often describes the distribution of averages.

### Estimation: Finding Model Parameters

#### Maximum Likelihood Estimation (MLE)

MLE is a method for finding the best parameters for a model given some observed data. It asks: "What parameter values make the data I observed the most likely?".

**Example: The Biased Coin** 
You flip a coin 10 times and get 7 heads and 3 tails. What is the best estimate for $p$, the probability of getting a head?
* MLE formalizes our intuition. It finds the value of $p$ that *maximizes* the probability of observing this specific outcome (7 heads, 3 tails).
* The math shows that the value that does this is exactly $p = \frac{7}{10} = 0.7$.

#### Maximum A Posteriori (MAP) Estimation

MAP is an extension of MLE that incorporates a *prior belief* about the parameter. It finds a balance between the evidence from the data (the likelihood) and your initial belief (the prior).

It asks: "Given the data AND my prior beliefs, what is the most probable parameter value?".

**Example:** If you were estimating the probability of heads for a coin, MLE might give you 0.7 from the data. But if you have a strong prior belief that most coins are fair (centered around 0.5), MAP will give you an estimate that is pulled slightly away from 0.7 and closer to 0.5.

### 🤖 Applications in Machine Learning

### Decision Boundary & Threshold

In classification, a model needs to make a decision (e.g., "is this a cat or a dog?").
* **Decision Boundary:** An invisible line or surface that separates the different classes. If a data point falls on one side, it's classified as "dog"; on the other, it's "cat".
* **Decision Threshold:** The specific cutoff value used to make the decision. For example, if a model outputs a probability, the threshold might be 0.5. If P(dog) > 0.5, classify as dog; otherwise, classify as cat.

The provided image shows two overlapping bell curves, representing the probability distributions for two different states (e.g., "signal present" vs. "no signal"). The vertical line is the decision threshold. The overlapping, shaded red area is the **Region of Error**, where the model is likely to make a mistake because the signals for the two states are ambiguous.

### Confusion Matrix

A table used to evaluate the performance of a classification model. It shows you not just *if* the model was wrong, but *how* it was wrong. For a binary (Yes/No) problem, it looks like this:

|                   | **Predicted: Positive** | **Predicted: Negative** |
| ----------------- | ----------------------- | ----------------------- |
| **Actual: Positive** | True Positive (TP)      | False Negative (FN)     |
| **Actual: Negative** | False Positive (FP)     | True Negative (TN)      |

* **True Positive (TP):** Correctly predicted positive (e.g., called spam spam).
* **True Negative (TN):** Correctly predicted negative (e.g., called a normal email normal).
* **False Positive (FP):** Wrongly predicted positive (Type I Error). A false alarm.
* **False Negative (FN):** Wrongly predicted negative (Type II Error). A miss.

From the confusion matrix, we can calculate key performance metrics:
* **Precision:** Of all the times the model predicted positive, how often was it right? $Precision = \frac{TP}{TP + FP}$.
* **Recall (Sensitivity):** Of all the actual positive cases, how many did the model find? $Recall = \frac{TP}{TP + FN}$.


### Quiz --> [Probability Theory Quiz](./Quiz/ProbabilityTheoryQuiz.md)

### Previous Topic --> [Linear Algebra](./LinearAlgebra.md)
### Next Topic --> [Optimization](./Optimization.md)

</div>