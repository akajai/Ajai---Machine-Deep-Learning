<div style="text-align: justify;">

## Bayesian Decision Theory

Bayesian Decision Theory is a statistical approach to pattern classification that quantifies the trade-offs between various classification decisions using probabilities and their associated costs. It operates under the assumption that all relevant probability distributions are known, aiming to find an optimal decision given these distributions. The core of this theory lies in Bayes' Rule, which allows for the conversion of prior probabilities (initial beliefs about an event) into posterior probabilities (updated beliefs after observing new evidence). This framework seeks to minimize the overall expected loss, also known as Bayes risk, by choosing the action that minimizes the conditional risk for each observation.

Bayes' Rule is a way of updating your beliefs based on new evidence. It's a mathematical formula that helps you figure out how likely something is to be true, given something else you've observed.

#### Applicability

- Fundamental statistical approach for pattern classification
- Quantifies trade-offs between classification decisions using probabilities and associated costs
- Assumption: Decision problem is posed probabilistically, and all relevant probabilities are known
- Optimal decision-making when underlying probability distributions are known

#### Formula

$$P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$$

- **P(A|B)**: This is the **conditional probability or posterior probability** of event A occurring given that event B is true. It represents the updated probability of A after observing B.
- **P(B|A)**: This is also a **conditional probability**, representing the likelihood of event B occurring given that event A is true. In the context of classification, this is often called the **likelihood**.
- **P(A)**: This is the **prior probability or marginal probability** of event A occurring without any given conditions. It reflects our initial knowledge or belief about A.
- **P(B)**: This is the **prior probability or marginal probability** of observing event B without any given conditions. In the context of the Bayes formula, it is sometimes referred to as evidence and acts as a scaling factor, ensuring that the posterior probabilities sum to one.

**Example**

Let's say you have a pet that seems unwell. You're trying to figure out if it has a specific rare disease.

- Belief (A): Your pet has the rare disease.
- Evidence (B): Your pet has a particular symptom (e.g., a cough).

Let's assign some imaginary probabilities:
- P(A) - The Prior: The disease is rare and affects only 1% of pets. So, your initial belief that your pet has the disease is low: P(A) = 0.01.
- P(B|A) - The Likelihood: If a pet does have this disease, it's very likely to have this cough. Let's say the probability of a sick pet coughing is 90%: P(B|A) = 0.90.
- P(B) - The Marginal Probability: The cough could be caused by many things, not just this rare disease. It could be a common cold, for instance. Let's say that overall, 10% of all pets (sick or healthy) will have this type of cough at any given time: P(B) = 0.10.

Now, let's plug these into Bayes' Rule to find P(A|B), the probability that your pet has the disease given that it has a cough.

$$P(A|B) = \frac{(0.90) * (0.01)}{(0.10)}$$

$$P(A|B) = \frac{0.009}{0.10}=0.09$$

So, even though your pet has a symptom that is very common in sick animals, the updated probability that it has the rare disease is only 9%. This is because the disease itself is so rare to begin with. Your initial belief (the prior) has a strong influence on the final outcome.

Bayes' Rule is a powerful tool used in many fields, from medical diagnoses and spam filtering to machine learning and weather forecasting, all to help make more accurate predictions when new information becomes available

#### Terminology

- **State of Nature ($\omega$)**: The true category or class label (e.g., $\omega_1$ for sea bass, $\omega_2$ for salmon)
- **Prior Probability ($P(\omega_j)$)**: Reflects initial knowledge about the likelihood of a state of nature before observation. Must exhibit exclusivity and exhaustivity ($P(\omega_1) + P(\omega_2) = 1$)
- **Decision Rule**: Prescribes an action based on observed input. A simple rule based only on priors is to decide $\omega_1$ if $P(\omega_1) > P(\omega_2)$, otherwise $\omega_2$.
- **Probability of Error (P(error))**: The chance of making an incorrect classification. When only prior probabilities are used, $P(error) = min[P(\omega_1), P(\omega_2)]$.
- **Feature (x)**: An observable variable used for classification (e.g., length, lightness)
- **Probability Density Function ($p(x)$ or $p(\mathbf{x})$)**: Represents the evidence of observing a pattern with feature value x
- **Class-Conditional Probability Density Function ($p(x|\omega_j)$)**: Also known as likelihood or state-conditional probability density. Describes the probability density of measuring a particular feature value x given that the pattern is in category $\omega_j$
- **Posterior Probability ($P(\omega_j|x)$)**: The probability of a certain state of nature given observed features. This is the quantity we want to use for decision-making after exploiting observation information

### Discriminant Functions

Imagine you're a judge at a talent show with multiple contestants. You can't just say "this one is good." You need a structured way to decide the winner.

So, you create a scoring system. You have a scorecard for each contestant where you assign points for different skills: singing, dancing, originality, etc. At the end, the contestant with the highest total score wins.

A discriminant function is exactly like that scorecard.

In Bayesian classification, when you have multiple categories (e.g., Cat, Dog, Bird), you create a separate discriminant function (a separate "scorecard") for each one. When you get a new piece of data (like a photo), you run it through every scorecard. The category whose scorecard gives the highest score is the winner.

In short: A discriminant function is a scoring function. The highest score wins.

### Decision Boundary

A decision boundary is the invisible line or surface where a classifier switches from choosing one category to another. It's the point of perfect indecision.

Think of it like drawing a border on a map. On one side of the line is Delhi, and on the other is Haryana. The decision boundary is that exact border line where you are no longer in one state and not yet in the other.
In Bayesian classification, this "border" is drawn where the posterior probabilities for two classes are exactly equal. A data point falling precisely on the boundary has a 50/50 chance of being classified as either category.

- A decision boundary is the border where a classifier is undecided between two or more classes.
- For normally distributed data, the shape of this boundary depends on the shape of the data clusters for each class
- If the clusters have the same shape, the boundary is a simple straight line
- If the clusters have different shapes, the boundary becomes a more complex curve.


### Probability of Error

In simple terms, the Probability of Error is the chance that your classifier makes a mistake. It's the likelihood that the system will assign the wrong label to an object.

Think of an email spam filter. Its job is to classify incoming emails into one of two categories: "Spam" or "Not Spam." An error can happen in two ways:

- False Positive: A legitimate email (like a bill or a message from a friend) is mistakenly put in the spam folder.
- False Negative: A junk email (like a fake lottery announcement) is mistakenly allowed into your main inbox.

The total Probability of Error is the combined chance of either of these mistakes happening. It's the overall "wrongness rate" of your classifier

#### Minimise the Error 

To minimize your chance of being wrong, the strategy is incredibly simple and intuitive: For any given situation, always choose the option that is most likely to be right.

This is known as the Bayes Decision Rule.

In the Bayesian world, "most likely to be right" means choosing the category with the highest posterior probability. Let's go back to our spam filter example to see how this works.

The Example: The Spam Filter's Decision

An email arrives. The filter needs to decide if it's Spam or Not Spam.

- The Evidence (x): The email's subject line is "You are a winner!"
- The Task: The filter calculates the posterior probability for both possible categories.
- Calculate P(Spam | "You are a winner!")
    - Based on its training, the filter knows that the words "winner" and "!" appear very frequently in junk mail. Therefore, the probability that this email is Spam, given these words, is very high.
    - Let's say the score is 98% (or 0.98).
- Calculate P(Not Spam | "You are a winner!")
    - The filter also knows that legitimate emails almost never use this phrase. So, the probability that this email is not Spam is extremely low.
    - Let's say the score is 2% (or 0.02).

Minimizing the Error:

The filter now compares the two scores:

Score(Spam) = 0.98

Score(Not Spam) = 0.02

The rule is to pick the highest score. Since 0.98 > 0.02, the filter chooses Spam.

By always, automatically, and instantly choosing the category with the highest posterior probability for every single email, the filter guarantees that it is making the fewest mistakes possible over the long run

### Feature Space

A feature space is an imaginary, multi-dimensional space where every piece of data is represented as a single point. Each dimension (or axis) in this space corresponds to a specific "feature" or characteristic of the data.
This concept allows us to translate abstract data into a geometric space, making it easier to find patterns, see relationships, and build classifiers.

Let's say we want to analyze apartments in Delhi. Our goal is to create a space where we can easily compare different properties.

**A One-Dimensional (1D) Space**

First, we choose just one feature: the price of the apartment. Our feature space is simply a number line. Every apartment is a single dot on this line based on its price.

This is useful, but limited. It only tells us about price, nothing else

**A Two-Dimensional (2D) Space**
To make it more useful, let's add a second feature: the size (in square feet). Now we have two dimensions.
- Axis 1 (X-axis): Size (sq. ft.)
- Axis 2 (Y-axis): Price (in Lakhs)

Our feature space is now a familiar 2D graph. Every apartment becomes a single point with (size, price) coordinates
- Apartment A: 800 sq. ft., ₹90 Lakhs → Plotted at point (800, 90)
- Apartment B: 1200 sq. ft., ₹150 Lakhs → Plotted at point (1200, 150)

In this 2D space, we can instantly see patterns. Apartments that are big and cheap (good deals) will be in the bottom-right. Apartments that are small and expensive (overpriced) will be in the top-left

**A Three-Dimensional (3D) Space and Beyond**

Let's add a third feature: distance from the nearest metro station (in km)

- Axis 1 (X-axis): Size
- Axis 2 (Y-axis): Price
- Axis 3 (Z-axis): Distance from Metro

Our feature space is now a 3D cube. Each apartment is a point floating in this cube. Real-world problems often use many more features (number of bedrooms, floor level, age of building, etc.), creating a high-dimensional space (a "hyperspace") that we can't visualize but can work with mathematically.

The whole point of creating a feature space is that similar items will cluster together.

Imagine we want to build a model to classify apartments as either "Good Deal" or "Overpriced".

- Plotting: We plot all known "Good Deal" and "Overpriced" apartments in our 2D feature space (Size vs. Price). We'll see two distinct clusters of points.
- Drawing Boundaries: The job of a classifier (like a Bayesian classifier) is to find and draw a decision boundary—a line or a curve—that best separates these two clusters
- Classifying New Data: When a new apartment listing appears, we simply calculate its coordinates and plot it in the space. The side of the decision boundary it falls on determines its classification.
In essence, feature space turns the complex problem of classification into a simpler geometric problem of separating groups of points.

### Conditional Risk

Conditional risk is the expected penalty or loss associated with taking a specific action, given a particular observation. It's the calculated "cost of being wrong" for a decision, based on the evidence you see right now.
To make this simple, think of a doctor making a diagnosis.

**An Example: The Doctor's Dilemma**

A patient arrives with a specific symptom: a persistent cough. The doctor needs to decide on a treatment plan

- The Observation (x): The patient has a persistent cough.
- The Possible True Conditions (States of Nature, ω):
    - The patient has a serious bacterial infection.
    - The patient has a common viral cold.
- The Possible Actions (α):
    - α1: Prescribe strong antibiotics.
    - α2: Recommend rest and fluids.

The doctor now calculates the conditional risk for each possible action, based on the evidence (the cough). This involves considering the "cost" (loss) of each outcome, weighted by its probability.

**Conditional Risk of Prescribing Antibiotics (R(α1|cough))**

The doctor thinks: "Given this cough, what's the chance it's bacterial vs. viral? And what's the penalty in each case if I prescribe antibiotics?"

- Scenario A: The cough is from a bacterial infection
    - Loss: 0 (Correct action, no penalty).
    - Probability: Given the cough, let's say the doctor estimates a 30% chance it's bacterial (P(Bacterial|cough) = 0.3).
- Scenario B: The cough is from a viral cold
    - Loss: 50 (Incorrect action. The penalty includes side effects, cost, and promoting antibiotic resistance).
    - Probability: Given the cough, there's a 70% chance it's viral (P(Viral|cough) = 0.7).

The conditional risk is the average expected loss:

Risk(Antibiotics|cough) = (Loss if Bacterial * Prob. of Bacterial) + (Loss if Viral * Prob. of Viral)

Risk(Antibiotics|cough) = (0 * 0.3) + (50 * 0.7) = 0 + 35 = 35

So, the expected penalty for this action is 35.

**Conditional Risk of Recommending Rest (R(α2|cough))**

The doctor repeats the process for the other action.

- Scenario A: The cough is from a bacterial infection.
    - Loss: 100 (Very wrong action! The penalty is high because the infection will get much worse without treatment).
    - Probability: 30% (P(Bacterial|cough) = 0.3).
- Scenario B: The cough is from a viral cold.
    - Loss: 0 (Correct action, no penalty).
    - Probability: 70% (P(Viral|cough) = 0.7).

The conditional risk for this action is:

Risk(Rest|cough) = (100 * 0.3) + (0 * 0.7) = 30 + 0 = 30

The expected penalty for this action is 30.

The doctor compares the conditional risk of every possible action:

- Risk of Prescribing Antibiotics = 35
- Risk of Recommending Rest = 30

To make the best decision, the doctor chooses the action with the lowest conditional risk. In this case, the decision is to recommend rest and fluids. Even though there's a chance it's a bacterial infection, the expected penalty for that mistake is lower than the expected penalty for prescribing unnecessary antibiotics

### Conditional Density 📊

A conditional density is a "profile" or "signature" that tells you what to expect for a given category.

Imagine we're building a system to sort fish on a conveyor belt into two types—Salmon and Sea Bass—based on a single feature: the lightness of their skin.

It answers the question: "If I already know this is a Salmon, what range of lightness values is typical?"

- p(lightness | Salmon): This is the conditional density for Salmon. We know from experience that Salmon are generally darker. So, this "profile" would look like a bell curve peaking over the darker shades. It tells us that it's highly probable to observe a dark fish if it's a Salmon.
- p(lightness | Sea Bass): This is the profile for Sea Bass. They are typically lighter. This profile would be a bell curve peaking over the lighter shades.

Essentially, it's a probability map for a feature, conditional on knowing the class.

### Likelihood Ratio ⚖️

The likelihood ratio measures the strength of your evidence. It compares how well the two profiles explain what you just saw.

It answers the question: "How many times more likely is this specific evidence if it came from Class A versus Class B?"

Let's say a new fish comes down the belt, and our sensor measures its lightness as a value of 12 (which is on the darker side). We now use our conditional densities to see how well each class explains this evidence:

- Looking at the Salmon profile, the probability of seeing a lightness of 12 is high. Let's say p(lightness=12 | Salmon) = 0.6.
- Looking at the Sea Bass profile, the probability of seeing a lightness of 12 is low. Let's say p(lightness=12 | Sea Bass) = 0.1.

The Likelihood Ratio is calculated as:

Likelihood Ratio=p(evidence∣Class B)p(evidence∣Class A)​=0.10.6​=6

This means the observed lightness of 12 is 6 times more likely to have come from a Salmon than a Sea Bass. It's a strong piece of evidence pointing towards Salmon.

### Threshold Value 🚧

The threshold is the "decision point" or the "bar you have to clear." It's the critical value you compare your likelihood ratio against to make the final call.

It answers the question: "Is my evidence strong enough to make a decision?"

The decision rule is:
- If Likelihood Ratio > Threshold, decide Class A.
- If Likelihood Ratio < Threshold, decide Class B.

This threshold isn't just a guess. It's carefully set based on two real-world factors:

1. Priors: How common is each fish? If Salmon are much rarer than Sea Bass, you'll need stronger evidence to say it's a Salmon. This would make the threshold higher.
2. Costs of Error: What's the penalty for a mistake? If misclassifying a Sea Bass as a Salmon is a very costly error (e.g., it ruins an expensive dish), you'd want to be extra sure. This would also raise the threshold, requiring overwhelming evidence before you dare to classify a fish as Salmon.

Let's say that after considering the costs and priors, our factory sets a threshold of 4.

The Final Decision:
- Our calculated Likelihood Ratio was 6.
- Our decision Threshold is 4.
- Since 6 > 4, our evidence is strong enough to clear the bar.
- The system decides to classify the fish as Salmon.

### Minimum error rate classification

Minimum error rate classification is a strategy where you always classify an observation into the category that is most probable, given the evidence. This simple and intuitive rule guarantees the lowest possible number of mistakes over the long run.
It's based on minimizing a specific type of penalty system called a zero-one loss function, where:
- The cost of a correct decision is 0.
- The cost of an incorrect decision is 1

Every mistake is treated as equally bad. The goal is to minimize the total number of "1s" you accumulate. The way to do that is to always bet on the most likely outcome.

Imagine you're designing an automated weather station for Delhi. Its only job is to predict if tomorrow will be "Sunny" or "Rainy." It makes this prediction based on today's atmospheric data.

- The Goal: To have the lowest possible error rate over the year.
- The Evidence (x): Today, the sensors measure low barometric pressure and high humidity.
- The Task: Based on this evidence, should the system predict "Sunny" or "Rainy" for tomorrow?

To achieve the minimum error rate, the system follows a simple two-step process based on historical data.

1. Calculate Posterior Probabilities

The system looks at its historical data and calculates the probability of each outcome, given the evidence:
- P(Rainy | low pressure, high humidity): "In the past, when we've seen this combination of low pressure and high humidity, what percentage of the time did it rain the next day?"
    - Let's say the answer is 85%.
- P(Sunny | low pressure, high humidity): "What percentage of the time was it sunny the next day?"
    - This would be 15%.
2. Apply the Minimum Error Rate Rule

The rule is simple: Choose the class with the highest posterior probability.
    - Probability of Rain: 85%
    - Probability of Sun: 15%

Since 85% > 15%, the system chooses "Rainy."

By making this choice, the system is accepting a 15% chance of being wrong. If it had chosen "Sunny," it would be accepting an 85% chance of being wrong. To minimize its error, it must go with the most probable outcome.

By consistently applying this "winner-take-all" logic for every day's evidence, the weather station ensures that its overall forecast accuracy is as high as it can possibly be. This is the essence of minimum error rate classification.

### Normal Density

A normal density, also known as a Gaussian distribution or the "bell curve," is a probability distribution that describes how data tends to cluster around a central average value.

It's a very common pattern found in nature and statistics. The main idea is that values near the average are very frequent, while values far from the average are increasingly rare.

- Bell Shape: The graph of the distribution looks like a symmetrical, bell-shaped curve
- Symmetry: The left and right sides of the curve are mirror images of each other
- Central Point: The mean (average), median (middle value), and mode (most frequent value) are all the same and are located at the exact center of the bell.
- Spread: The width of the bell is determined by the standard deviation. A small standard deviation results in a tall, narrow curve (data is tightly packed), while a large standard deviation results in a short, wide curve (data is spread out).

**Example: Dosa Preparation Time**
Imagine a popular restaurant in Delhi that makes hundreds of dosas every day. If you measure the time it takes for the chef to prepare each dosa, you'll likely find a normal distribution:
- Most Frequent Time (The Mean): Most dosas will take around the average time to prepare, let's say 3 minutes. This is the peak of the bell curve.
- Less Frequent Times: It will be slightly less common for a dosa to take 2.5 minutes or 3.5 minutes.
- Rare Times: It will be very rare for a dosa to be made in under 1 minute or take longer than 5 minutes.

If you plot all these preparation times on a graph, you would get the classic bell shape, showing that extreme values (very fast or very slow) are much rarer than average values. This pattern appears for countless real-world phenomena, such as the heights of people, exam scores, and measurement errors.

### Univariate Density

A univariate density describes the probability distribution of a single variable. The prefix "uni" means one, so you're looking at the likelihood of each possible outcome for just one thing at a time.

The normal density (bell curve) is a very common type of univariate density.

**An Example: Analyzing a Cricketer's Scores**

Imagine you're analyzing the performance of a star batsman from the Delhi Capitals. You collect the scores from his last 100 matches. The single variable you are interested in is runs scored.

If you create a graph showing how frequently he achieved each score, that graph represents the univariate density of his runs.

- X-axis: Runs Scored (e.g., 0, 1, 2, ... 100+)
- Y-axis: Probability or Frequency of that score

This graph might show you:

- A high probability of scores between 20 and 50 runs (his typical range).
- A lower probability of him scoring a century (100+ runs), as this is a rare event.
- A certain probability of him getting out for a duck (0 runs).

You are analyzing only one feature—the runs. You are not yet looking at how his score relates to the stadium, the opponent, or the time of day. You are simply describing the distribution of that single variable, which is the essence of a univariate density.

### Multivariate Density

A multivariate density describes the joint probability distribution of multiple variables at the same time. It tells you the likelihood of observing a specific combination of values across all those variables simultaneously.

While a univariate density looks at one feature in isolation, a multivariate density looks at the complete picture and the relationships between features.

**An Example: Analyzing Delhi's Weather**

Let's say we want to analyze the weather in Delhi using two variables at once: Temperature and Humidity. A multivariate density will tell us the probability of observing a specific pair of these values on any given day.

- Univariate: "What's the probability that the temperature is 40°C?"
- Multivariate: "What's the probability that the temperature is 40°C and the humidity is 75% at the same time?"

**Visualizing the Density**

For two variables, we can't use a simple curve. We need a 3D surface, like a landscape map
- The ground is our 2D feature space (Temperature on one axis, Humidity on the other).
- The height of the landscape at any point represents the probability density.

This "probability landscape" for Delhi's weather might look like this:

- A High Mountain Peak: There would be a tall peak over the combination of (High Temperature, High Humidity). This represents the monsoon season (July-August), a very common weather pattern.
- Another High Ridge: There might be another peak or high ridge over (High Temperature, Low Humidity), representing the hot, dry pre-monsoon season (May-June).
- Deep Valleys: The landscape would be very low (a deep valley) over combinations like (Low Temperature, High Humidity), as this is a very rare weather pattern in Delhi.

This 3D map is the multivariate density. It doesn't just tell us about typical temperatures or typical humidity levels; it tells us about the typical combinations and how the two variables interact.


### Covariance Matrix

A covariance matrix is a square grid of numbers that summarizes the variance and relationships of multiple variables in a dataset. It tells you two key things:

1. How spread out each variable is on its own (Variance).
2. How each pair of variables changes together (Covariance).

Let's use a simple example with two variables for apartments in Delhi: Size (in sq. ft.) and Price (in Lakhs). The covariance matrix would be a 2x2 grid.

1. The Diagonal Elements: Variance

    The numbers on the main diagonal (from top-left to bottom-right) represent the variance of each individual variable. Variance is a measure of how spread out the data is.
    - Top-Left (Var(Size)): A large number here means apartment sizes in your dataset vary a lot (from tiny studios to huge penthouses). A small number means they are all roughly the same size.
    - Bottom-Right (Var(Price)): A large number here means the prices are widely spread out.

2. The Off-Diagonal Elements: Covariance
    All other numbers in the grid represent the covariance between pairs of variables. Covariance measures how two variables move in relation to each other.
     - Positive Covariance (Cov(Size, Price) > 0): This means that as one variable increases, the other tends to increase. For our example, this value would be positive because as the size of an apartment goes up, its price also tends to go up.
     - Negative Covariance: This would mean that as one variable increases, the other tends to decrease (e.g., as the age of a car increases, its price decreases).
     - Zero Covariance: This means the two variables have no linear relationship (e.g., apartment size and the number of rainy days in a year).

The covariance matrix is always symmetric because the relationship between Size and Price is the same as the relationship between Price and Size.

In machine learning and statistics, the covariance matrix is crucial because it describes the shape and orientation of your data cloud in the feature space. This shape, in turn, helps determine the optimal decision boundary for a classifier. For a multivariate normal distribution, the covariance matrix defines whether the data forms a circular cluster, a stretched-out ellipse, or a tilted ellipse.

### Independent Binary Features

Independent binary features are simple "yes/no" characteristics of your data that are treated as being completely unrelated to one another. The state of one feature gives you no clues about the state of another.

- Binary: The feature has only two states (e.g., yes/no, true/false, 1/0).
- Independent: The features don't influence each other.

**An Example: Ordering a Pizza 🍕**

Imagine you're analyzing pizza orders. You decide to track two features for each order:
- Feature A (Binary): Did the customer order extra cheese? (Yes/No)
- Feature B (Binary): Was the pizza delivered? (Yes/No, as opposed to takeaway)

For these two features to be considered independent, we must assume that a customer's decision to order extra cheese has absolutely no connection to their decision to have the pizza delivered.

Under this assumption, the probability of someone wanting extra cheese is the same whether they are dining in, taking it away, or getting it delivered. The two choices are treated as separate, unrelated events.

In reality, there might be a subtle link (a dependency). Perhaps customers who get delivery are slightly more likely to splurge on extra cheese. However, for a simple model like Naive Bayes, we make the "naive" assumption that they are independent. This allows the model to calculate the total probability by simply multiplying the individual probabilities of each feature, making the math much easier.

**A Counter-Example: Dependent Features**

Now consider these two features for a car:

- Has a sunroof? (Yes/No)
- Is it an automatic transmission? (Yes/No)

These features are not independent. If you know a car is an automatic, the probability that it also has a sunroof is much higher than for a manual car, because both are often features of higher-end models. Knowing one feature helps you predict the other.

### Recall 

Recall is a performance metric that measures a model's ability to find all the actual positive cases within a dataset. It's a crucial concept when the cost of missing a positive case is very high.

The term "recall discriminant function" isn't standard. The correct way to think about it is how a discriminant function's decision rule can be adjusted to prioritize recall.

In simple terms, recall answers the question: "Of all the things we were supposed to find, what percentage did we actually find?"

It's a measure of completeness. A high recall means you are not missing many positive cases.

Formula:

$$Recall = \frac{(True Positives)} {(True Positives + False Negatives)}$$

A False Negative is an event you were supposed to find but missed—this is what recall aims to minimize.

Example: A Hospital's Cancer Screening AI

- Positive Case: A patient who actually has cancer.
- The Goal: The hospital wants to identify every single patient who has cancer.
- A False Negative (The Worst Outcome): The AI says a sick patient is healthy. The patient is sent home, and the disease progresses untreated. This is a catastrophic error.
- A False Positive: The AI flags a healthy patient as potentially sick. This leads to more tests and anxiety, which is not ideal, but it's far better than the alternative.

For this task, recall is the most important metric. The hospital wants the AI's recall to be as close to 100% as possible, even if it means having some false alarms.


### Naïve Bayes Theory

Naive Bayes is a simple and popular classification algorithm based on Bayes' Theorem. It's used to predict the category of an object based on its features.

The algorithm is called "naive" because it operates on one big, simplifying, and often unrealistic assumption.

**Key Assumption**
- The key assumption of Naive Bayes is that all features are independent of one another, given the class
- In simple terms, this means the algorithm assumes that the presence of one feature has absolutely no bearing on the presence of any other feature. It treats every piece of information as a completely separate and unrelated clue
- This assumption is "naive" because in the real world, features are very often related. For example, in an email, the word "sale" is more likely to appear if the word "discount" is also present. Naive Bayes simply ignores this connection.

**Example: Is it a Dog or a Cat? 🐶 vs 🐱**

Let's say we're building a simple classifier to identify an animal as either a Dog or a Cat.

- The Animal (Evidence): We observe an animal that barks, is friendly, and hates baths
- The Task: Is it more likely a Dog or a Cat?

A Naive Bayes classifier would calculate the probability of it being a Dog and the probability of it being a Cat, and then pick the higher one. Here’s how the naive assumption simplifies its thinking:

Instead of trying to figure out the complex probability of an animal that barks, is friendly, and hates baths all at once, it breaks the problem down. It assumes these three features are totally unrelated.

1. It evaluates P(Dog | barks, friendly, hates baths) by asking separately:
    - What's the probability that a Dog barks? (Very High)
    - What's the probability that a Dog is friendly? (High)
    - What's the probability that a Dog hates baths? (High)
    - It then multiplies these high probabilities together to get a final high score for "Dog."
2. It evaluates P(Cat | barks, friendly, hates baths) by asking separately:
    - What's the probability that a Cat barks? (Extremely Low)
    - What's the probability that a Cat is friendly? (Medium)
    - What's the probability that a Cat hates baths? (Very High)
    - When it multiplies these probabilities, the extremely low probability of "barks" will make the final score for "Cat" incredibly low, regardless of the other features

The final score for Dog will be much higher than the score for Cat. The classifier will confidently label the animal as a Dog.

**Types of Navie Bayes Algorithms**

There are three main types of Naive Bayes algorithms: Gaussian, Multinomial, and Bernoulli. Each is designed to handle a different kind of data.

#### Gaussian Naive Bayes

This is the version to use when your features are continuous numerical data that you can assume follows a normal distribution (a "bell curve").

- **Type of Data**: Continuous values like height, weight, temperature, or any real-world measurement
- **Analogy**: It's like a tailor identifying a person's athletic background. The tailor assumes that the measurements (height, arm length, waist size) for swimmers, runners, and weightlifters each follow a bell curve. When a new person comes in, the tailor checks whose statistical "profile" their measurements fit best

**Example: Classifying Iris Flowers**

Imagine you need to classify a flower as one of three species of Iris based on its measurements.
- The Task: Classify a flower as 'Setosa', 'Versicolor', or 'Virginica'
- The Features (Continuous): Petal Length (cm), Petal Width (cm), Sepal Length (cm)
- How it Works: The algorithm first calculates the average (mean) and spread (standard deviation) of petal length, petal width, etc., for each of the three flower species from a training dataset. When you find a new flower, it calculates the probability of its specific measurements belonging to each species' bell curve. The species with the highest overall probability is the winner

#### Multinomial Naive Bayes

This is the go-to version for features that represent counts or frequencies. It's most famously used in text classification
- **Type of Data**: Discrete counts. For example, the number of times a word appears in an email
- **Analogy**: It's like a librarian sorting books onto shelves. The librarian knows that books on the 'History' shelf have a high frequency of words like "war" and "king," while 'Science' books have a high frequency of "energy" and "matter." When a new book arrives, the librarian quickly scans its word counts to decide which shelf it most likely belongs on.

**Example: Classifying News Articles**
You want to automatically categorize news articles into 'Sports', 'Politics', or 'Technology'
- The Task: Categorize a news headline like "Local team wins championship game."
- The Features (Counts): The frequency of each word. (e.g., 'team': 1, 'wins': 1, 'game': 1)
- How it Works: The model learns the typical word counts for each category. 'Sports' articles will have a high count of words like "team," "win," "score," and "ball." When the new headline comes in, the algorithm calculates the probability that this specific "bag of words" belongs to the 'Sports', 'Politics', or 'Technology' category. In this case, it would be a strong match for 'Sports'.

#### Bernoulli Naive Bayes

This is a simpler version used when your features are binary—meaning they are either "present" or "absent." It doesn't care how many times a feature appears, only if it's there or not

- **Type of Data**: Binary (yes/no, true/false, 1/0)
- **Analogy**: It's like deciding if a restaurant is good based on a checklist of buzzwords in its reviews. Your checklist might include "delicious," "amazing," "rude," and "terrible." You only care if a review contains the word "amazing" (yes/no), not if it contains it five times.

**Example: Filtering Spam Email**
You want to classify an email as 'Spam' or 'Not Spam'

- The Task: Classify an email based on the presence or absence of certain keywords.
- The Features (Binary): A predefined list of words. For an incoming email, each feature is either 1 (the word is present) or 0 (the word is absent)
    - contains_winner: 1
    - contains_free: 1
    - contains_invoice: 0
    - contains_meeting: 0
- How it Works: The algorithm learns the probability of these words appearing in spam versus non-spam emails. The presence of "winner" and "free" strongly suggests an email is spam. It combines the probabilities from all the binary features to make a final decision.


### Quiz --> [Quiz](./Quiz/BayesianDecisionTheoryQuiz.md)

### Previous Topic --> [Linear Regression](./LinearRegressionModel.md)


</div>