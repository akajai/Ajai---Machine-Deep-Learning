<div style="text-align: justify;">

## Decision Tree

A Decision Tree is a supervised, non-parametric machine learning model that predicts an outcome by learning simple decision rules from data. Think of it as a highly organized flowchart of "if-then-else" questions that the computer creates by itself to arrive at a conclusion.

Imagine you want to decide whether to play tennis today. Your decision depends on a few factors: the weather outlook, the humidity, and the wind. A Decision Tree is like your internal thought process for making this choice.

You might ask yourself a series of questions:

1. Is the weather outlook sunny, overcast, or rainy?
    - If it's rainy, you decide not to play. (This is a final decision).
    - If it's sunny, you move to the next question.
    - If it's overcast, you decide to play. (This is another final decision).
2. For the sunny path, you then ask: Is the humidity high or normal?
    - If it's high, you decide not to play.
    - If it's normal, you decide to play.

This entire structure is a Decision Tree.
- The first question ("Weather Outlook?") is the root node.
- The links between questions ("Sunny," "Rainy") are the branches.
- The subsequent questions ("Humidity?") are internal nodes.
- The final outcomes ("Play" or "Don't Play") are the leaf nodes.

A Decision Tree doesn't assume any specific relationship between the inputs (weather, humidity) and the output (play tennis?). It builds the rules based on what it finds in the data.

### How Does a Decision Tree "Learn"?

The magic of a Decision Tree is how it automatically figures out the best questions to ask and in what order. The main goal is to split the data into groups that are as "pure" as possible at each step.

Imagine you have a basket of 10 red and 10 blue balls all mixed up. This basket is "impure" or very mixed. You want to ask a question that helps you separate them.

- Bad Question: "Is the ball's serial number odd?" This probably won't help separate the colors well. You'd still have a mix of red and blue balls in both the "yes" and "no" piles.
- Good Question: "Is the ball bigger than 2 inches?" If all the red balls are large and all the blue balls are small, this one question perfectly separates them into two "pure" groups—one with only red balls and one with only blue balls.


A Decision Tree algorithm does exactly this. It looks at all the possible questions (features like "Outlook," "Humidity") and picks the one that results in the purest, most separated groups. It keeps doing this at each level until it reaches a conclusion (a leaf node) or runs out of questions to ask. The measures it uses to quantify "purity" have technical names like Gini Impurity or Entropy, but they are just mathematical ways of measuring how mixed-up a group is.

**Pros of Decision Trees 👍**

- Easy to Understand and Interpret: Their flowchart-like structure is very intuitive. You can easily visualize and explain the model's decision-making process to non-technical people.
- Requires Little Data Preparation: Unlike many other models, they don't require you to normalize or scale your numerical data.
- Handles Different Data Types: They can work with both numerical data (e.g., temperature) and categorical data (e.g., sunny, overcast, rainy) seamlessly.
- White Box Model: The reasoning behind a prediction is clearly visible, unlike "black box" models (like complex neural networks) where the internal logic is hidden.

**Cons of Decision Trees 👎**

- Overfitting: This is their biggest weakness. A tree can become overly complex and learn the noise and quirks of the training data too perfectly. When this happens, it performs poorly on new, unseen data. It's like a student who memorizes the exact answers for a practice test but fails the actual exam because they didn't learn the underlying concepts. This can be controlled by "pruning" the tree (snipping off some branches).
- Instability: A small change in the training data can sometimes lead to a completely different tree being generated.
- Biased with Imbalanced Data: If one outcome is much more frequent than another in your dataset, the tree will be biased towards predicting the more frequent outcome.

**Key Terminology**

- Root Node: The very top node that represents the entire dataset of loan applicants. For example, 1000 past applications, of which 600 were approved and 400 were denied.
- Splitting: The process of dividing a node into two or more sub-nodes. The algorithm looks at all features (like 'Credit Score', 'Income', 'Age') and chooses the one that creates the "purest" groups.
- Decision Node: A node that splits further. For example, the root node might split based on the question: "Is Credit Score > 700?". This creates two branches: one for applicants with scores above 700 and one for those below. This node represents a test on a feature.
- Leaf/Terminal Node: The end of a branch. It represents a final decision (a class label) and does not split any further. For example, a leaf node might be "Approve Loan" because it contains applicants who all have high credit scores and high incomes.
- Purity (Gini Impurity or Entropy): This is a measure of how mixed up the results are in a single node. A pure node would contain applicants who were all approved (or all denied). An impure node has a mix. The goal of splitting is to find a question that leads to purer child nodes. For instance, splitting by credit score might create one group with 90% "Approve" and another with 80% "Deny"—both are purer than the original 60/40 mix.
- Pruning: This is the process of removing branches from the tree to prevent it from becoming too complex and overfitting the training data. A tree that is too detailed might not perform well on new, unseen loan applications.

### Depth

In the context of a decision tree, depth refers to the length of a path from the root node to a specific node, while max depth is a hyperparameter that limits the maximum possible depth for the entire tree.

The depth of a node is the number of edges or steps you have to take to get from the root node (the very top of the tree) to that specific node.

- The root node itself is always at Depth 0.
- The nodes directly connected to the root are at Depth 1.
- Their children are at Depth 2, and so on.

Think of it like floors in a building. The ground floor is Depth 0, the first floor is Depth 1, etc. The depth tells you how many questions (or splits) have been asked to reach that point in the decision-making process.

### Max Depth

Max depth is a setting or hyperparameter that you define before training your decision tree.3 It puts a limit on how deep the tree can grow.4 It's the length of the longest possible path from the root node to any leaf node in the tree.

If you set max_depth=3, your tree can ask at most three levels of questions before it is forced to make a final prediction.

The value you choose for max_depth has a significant impact on your model's performance:

- Low Max Depth (e.g., 2 or 3): This creates a "shallow" tree with very few decision rules.
    - Pros: The model is simple, fast, and easy to interpret.6 It's less likely to be influenced by noise in the data.7
    - Cons: It might be too simple and fail to capture important patterns, leading to underfitting (high bias).8
- High Max Depth (e.g., 10 or more): This allows for a "deep" tree with many complex decision rules.9
    - Pros: The model is very flexible and can capture intricate relationships in the data.
    - Cons: It is highly likely to overfit.10 This means it essentially memorizes the training data, including its noise and quirks, and will perform poorly on new, unseen data (high variance).

Max depth is one of the most important tools for controlling the complexity of a decision tree and preventing overfitting.12 It helps you manage the bias-variance trade-off:

- A shallow tree is simple (high bias) but stable (low variance).
- A deep tree is complex (low bias) but unstable (high variance).

The goal is to find a max_depth that allows the model to learn the true underlying patterns in the data without memorizing its noise.13 This "sweet spot" is typically found using techniques like cross-validation, where you test different values for max_depth to see which one performs best on data it hasn't seen before.

### Type of Decision Tree

#### Classification Decision Tree

A classification decision tree is a predictive model that uses a tree-like structure of decisions to classify data into specific categories. It works by asking a series of simple if-then-else questions about the data's features to narrow down the possibilities until it arrives at a final conclusion or class label.

**Fruite Game**

Imagine you're playing a guessing game. Your friend is thinking of a fruit, and your goal is to guess what it is by asking yes/no questions. The fruit can only be an Apple or an Orange.

Your questioning strategy might look like this:

1. Question 1: "Is the color red?"
    - If YES, you guess: Apple.
    - If NO, you ask another question.
2. Question 2: "Is the diameter bigger than 3 inches?"
    - If YES, you guess: Orange.
    - If NO, you guess: Apple.

You've just created a simple classification decision tree. It takes features (color, diameter) and follows a path of decisions to predict a category (Apple or Orange).

**Email Spam**

Let's see how a computer would build a tree to classify an email as "Spam" or "Not Spam". The model learns from a dataset of past emails that have already been labeled.

Objective: Predict if a new email is Spam or Not Spam.

Features:

- Does it contain the word "lottery"? (Yes/No)
- Is the sender in your contacts? (Yes/No)
- Does it have attachments? (Yes/No)

1. Step 1: Find the Best First Question (The Root Node)

    The algorithm looks at all the features and asks: "Which single question does the best job of splitting our pile of emails into two groups, where one group is mostly Spam and the other is mostly Not Spam?"

    This "best split" is determined mathematically using a measure of purity, like Gini Impurity or Entropy. A pure group has only one class in it (e.g., all Spam). The algorithm found that the question "Does it contain the word 'lottery'?" is the most effective first split.

    This becomes the root node.
2. Step 2: Split the Data and Repeat

    The emails are now divided into two branches:

    - Branch A (Yes to 'lottery'): This group is very likely to be Spam. Let's say 95% of these emails are Spam. Since this group is almost pure, we can stop here and create a leaf node with the prediction: "Spam".
    - Branch B (No to 'lottery'): This group is a mix of Spam and Not Spam. The tree needs to ask another question to sort this group out.

    The algorithm now focuses only on Branch B and repeats Step 1. It looks at the remaining features ('Sender in contacts?', 'Has attachments?') and finds the next best question. Let's say it's "Is the sender in your contacts?". This becomes a decision node.

3. Step 3: Continue Splitting Until You Reach an Answer

    The process continues until the tree can't make any more useful splits. A branch stops splitting and becomes a leaf node when:

    - The group is pure (or mostly pure).
    - A pre-set limit is reached (like a maximum depth).
    - The group is too small to be split further.

    The final tree might look like this:
4. How the Tree Makes a Prediction

    Now, a new, unlabeled email arrives. To classify it, you simply run it down the tree from top to bottom:

    - Start at the Root: Does the email contain the word "lottery"? No.
    - Move to the Next Node: Follow the "No" branch. Is the sender in your contacts? No.
    - Arrive at a Leaf Node: Follow the "No" branch. The leaf node says: "Spam".

    The model's final prediction for this new email is Spam.

#### Regression Decision Tree

A regression tree is a powerful and intuitive machine learning model that predicts a continuous numerical value, like a price, age, or temperature. It works by building a tree-like structure of if-then-else questions to segment the data into groups with similar outcomes.

**A Smart Price Estimator**

Imagine you're trying to estimate the price of a used car. You don't just guess randomly; you ask a series of questions to narrow down the price range.

1. First Question: "Is the car's mileage over 100,000 miles?"

    - If YES, you know it's in a lower price bracket, maybe averaging around $8,000.
    - If NO, it's in a higher price bracket, maybe averaging $20,000. You need more information.
2. For the lower-mileage cars, you ask: "Is the car a luxury brand?"

    - If YES, the average price might jump to $28,000.
    - If NO, the average might be around $16,000.

You've just performed the logic of a regression tree. You split your data based on features to arrive at groups with similar, predictable average values.

**House Price Example**

Let's walk through how a computer would build a regression tree to predict house prices. The model learns from a dataset of past house sales.

Objective: Predict the price of a house (a continuous number).

Features:

- Square Footage (e.g., 2,100 sq ft)
- Number of Bedrooms (e.g., 3)
- Neighborhood (e.g., A, B, or C)

1. Step 1: Find the Best First Split (The Root Node)

    Unlike a classification tree that tries to create "pure" categories, a regression tree tries to create groups where the house prices are as similar as possible. It wants to minimize the variance within the groups.

    The algorithm measures this using a metric like Mean Squared Error (MSE). It searches every possible split on every feature to find the one that results in the lowest MSE.

    - It might test: Square Footage > 2000?
    - It might test: Number of Bedrooms > 3?
    - It might test: Neighborhood is A?

    Let's say the algorithm finds that splitting the data by Square Footage > 2000 creates two groups where the prices inside each group are much less spread out than the original dataset. This becomes the root node.

2. Step 2: Split the Data and Repeat

    The houses are now divided into two branches:

    - Branch A (<= 2000 sq ft): The average price here might be $250,000.
    - Branch B (> 2000 sq ft): The average price here might be $450,000.

    The algorithm now treats each branch as a separate problem and repeats Step 1. For Branch B (the larger houses), it might find that the next best question is "Is the Neighborhood A?". This creates another split, further segmenting the houses.

3. Step 3: Stop Splitting and Create Leaf Nodes

    A branch stops growing when a stopping condition is met, such as a pre-set maximum depth or when a node has too few houses to make a meaningful split. These final nodes are the leaf nodes.

4. How the Tree Makes a Prediction

    This is the most important part. The prediction for any new data point is simply the average of the target values of all the training data that ended up in that leaf.

    Let's say a new house comes on the market, and we want to predict its price:

    - Features: 2,500 sq ft, 4 Bedrooms, Neighborhood A.

    We run it through the tree:

    - Start at the Root: Is the square footage > 2000? Yes. Follow that branch.
    - Move to the Next Node: Is the neighborhood A? Yes. Follow that branch.
    - Arrive at a Leaf Node: This branch ends in a leaf. The tree looks at all the houses from its training data that also ended up in this leaf. Let's say there were 10 such houses, and their average price was $525,000.

    The model's final prediction for the new house is $525,000.


### Greedy Recursive Process

A greedy recursive process is a method for solving a problem by breaking it into smaller, identical versions of itself (recursion) and, at each step, making the choice that seems best at that immediate moment (greedy).

It doesn't plan ahead; it just makes the optimal local choice at every stage, hoping it will lead to the best overall solution.

**The "Greedy" Part 🍰**

The "greedy" part means making the choice that offers the biggest immediate reward. Imagine you're at a buffet and you take the biggest piece of cake first without looking at all the other desserts. You're optimizing for the current step, not the entire meal.

**The "Recursive" Part nesting dolls**

The "recursive" part means the process breaks a big problem down into smaller versions of the exact same problem. Once it solves a small piece, it uses that solution to help solve the bigger problem. It's like a set of Russian nesting dolls; to understand the whole set, you open one doll to find a smaller, identical doll inside, and you repeat the process.

**The Change-Making Example 💰**

Let's say you're a cashier and need to give a customer 78 cents in change using the fewest coins possible (quarters: 25¢, dimes: 10¢, nickels: 5¢, pennies: 1¢).

A greedy recursive process would work like this:

1. The Problem: Give 78¢.

    - Greedy Choice: What's the biggest coin I can use without going over? A quarter (25¢).
    - Recursive Step: Now I have a new, smaller problem: give the remaining 53¢ (78 - 25).
2. The Problem: Give 53¢.

    - Greedy Choice: What's the biggest coin I can use? Another quarter (25¢).
    - Recursive Step: Now I have a new problem: give the remaining 28¢ (53 - 25).
3. The Problem: Give 28¢.

    - Greedy Choice: A quarter is too big. The next biggest is a dime (10¢).
    - Recursive Step: New problem: give the remaining 18¢ (28 - 10).
4. The Problem: Give 18¢.

    - Greedy Choice: Another dime (10¢).
    - Recursive Step: New problem: give the remaining 8¢ (18 - 10).
5. The Problem: Give 8¢.

    - Greedy Choice: A dime is too big. The next is a nickel (5¢).
    - Recursive Step: New problem: give the remaining 3¢ (8 - 5).
6. The Problem: Give 3¢.

    - Greedy Choice: A nickel is too big. I'll use a penny (1¢).
    - Recursive Step: And again, and again, until the amount is 0.

The process stops when the problem is solved (the amount is zero). The final solution is 2 quarters, 2 dimes, 1 nickel, and 3 pennies. In this case, the greedy approach worked perfectly.

Building a decision tree is a classic example of a greedy recursive process.

- Recursive: The algorithm's main task is "build a tree for this dataset." After it makes the first split, it runs the exact same task on the two smaller datasets created by the split.
- Greedy: At each node, the algorithm looks at every possible feature to split on. It greedily chooses the single best split—the one that creates the purest groups at that moment. It doesn't look ahead to see if a less-optimal split now could lead to an even better tree two or three levels down the line. It just takes the best immediate win it can find.

### Selecting the feature Split

A decision tree selects the best feature to split on by trying every possible split for every feature and choosing the one that results in the "purest" subgroups. In simple terms, it looks for the single question that does the best job of separating the data into the most homogeneous groups possible

Imagine you have a group of data points. If the group is very mixed in its outcomes (e.g., 50% approved loans and 50% denied loans), it has high impurity or high uncertainty. The primary goal of a split is to ask a question that reduces this uncertainty.
A good split will create new groups (called child nodes) that are "purer" or more uniform than the original group (the parent node).

The tree uses specific mathematical formulas to quantify the "goodness" of a split.

### Information Gain

Information Gain is the measure of how much a feature split reduces uncertainty. It's calculated by measuring the level of disorder, called Entropy, before a split and subtracting the disorder remaining after the split. A feature split with a higher Information Gain is better because it does a more effective job of organizing the data.

How much did this feature split help us reduce the randomness and create more organized, purer groups?

Information Gain = Entropy(parent) - Weighted Average Entropy(children)

A decision tree uses this value to select the best feature to split on. It calculates the Information Gain for every possible split and chooses the one with the highest score.


### Entropy

Entropy is a measure of randomness, impurity, or uncertainty.

Imagine you have a deck of cards.

- Low Entropy (Value = 0): The deck is perfectly sorted by suit and number. There is no randomness. If you pick a card, you have a very good idea of what the next one will be. This is a "pure" state.
- High Entropy (Value = 1): The deck is perfectly shuffled. There is maximum randomness. You have no idea what the next card will be. This is an "impure" state.

In a dataset, a group of data points has low entropy if most of them belong to the same class (e.g., a group where 95% of people clicked "Buy"). It has high entropy if the classes are mixed evenly (e.g., 50% clicked "Buy" and 50% did not)

Two-class entropy is a measure of impurity or uncertainty in a group of data where every item belongs to one of only two categories (e.g., Yes/No, Spam/Not Spam, True/False). It's a score between 0 and 1 that tells you how mixed that group is.

- Entropy = 0: This means the group is perfectly pure.1 All items belong to a single class. There is zero uncertainty.2
- Entropy = 1: This means the group is perfectly mixed (a 50/50 split). This is the state of maximum uncertainty.

The easiest way to understand two-class entropy is to think about flipping a coin.

- Zero Entropy: Imagine you have a two-headed coin. Every flip will be "Heads." The outcome is 100% certain. This group of flips has an entropy of 0.
- Maximum Entropy: Now imagine a fair coin. There is a 50% chance of "Heads" and a 50% chance of "Tails." You have the least amount of information possible about the next flip's outcome. This is maximum randomness, so the entropy is 1.3
- Medium Entropy: What about a weighted coin that lands on "Heads" 90% of the time? The group of flips is still impure, but you're pretty sure the next one will be "Heads." There's only a little uncertainty. This group would have a low entropy, somewhere close to 0 (specifically, 0.47).


The formula for two-class entropy looks a bit intimidating, but the concept is simple:

Entropy(S)=−p1​log2​(p1​)−p2​log2​(p2​)

- p1​: The proportion of items in Class 1 (e.g., the percentage of emails that are "Spam").
- p2​: The proportion of items in Class 2 (e.g., the percentage of emails that are "Not Spam").
- log2​(p): This part of the formula represents the "surprise" or amount of information in an event. A very rare event (low p) is very surprising. A very common event (high p) is not surprising at all.

The formula calculates the weighted average "surprise" for the entire group. When uncertainty is highest (a 50/50 split), the average surprise is maximized.4 When uncertainty is lowest (a 100/0 split), there is no surprise at all.

Imagine a group of 10 emails.

- Case 1: 10 are "Spam," 0 are "Not Spam." (p₁=1.0, p₂=0.0) -> Entropy = 0.
- Case 2: 7 are "Spam," 3 are "Not Spam." (p₁=0.7, p₂=0.3) -> Entropy = 0.88. (High impurity)
- Case 3: 5 are "Spam," 5 are "Not Spam." (p₁=0.5, p₂=0.5) -> Entropy = 1.0. (Maximum impurity)

**Entropy Behaviour**

Entropy measures the level of uncertainty or impurity in a group of data. Its behaviour is governed by one main principle: it increases as randomness increases and decreases as uniformity increases.

In simple terms, entropy is lowest (zero) when a group is perfectly pure (contains only one class) and is highest when the classes are mixed as evenly as possible.

### Gini Impurity

Gini Impurity is a score that measures the level of impurity or "mixed-up-ness" in a group of items. In simple terms, it calculates the probability of incorrectly classifying a randomly chosen item from the group if you were to label it randomly according to the distribution of labels within that group.

A lower Gini Impurity score is better, as it signifies a "purer" group.

- Gini = 0: This means the group is perfectly pure. All items belong to a single class.
- Gini = 0.5: (for a two-class problem) This means the group is perfectly mixed (a 50/50 split), representing maximum impurity.

Imagine you have a jar of marbles. The Gini Impurity score tells you the chance of messing up if you play a simple guessing game.

- Scenario 1: Low Gini Impurity

    - The jar contains 10 red marbles and 0 blue marbles.
    - The group is perfectly pure. If you pick a marble and guess its color, you'll always guess "red" because that's all that's in there. Your chance of being wrong is 0%.
    - Gini Impurity = 0.
- Scenario 2: High Gini Impurity

    - The jar contains 5 red marbles and 5 blue marbles.
    - The group is perfectly mixed. If you pick a marble and have to guess its color, you have a 50% chance of being wrong. This is the highest possible level of impurity.
    - Gini Impurity = 0.5.

Gini Impurity essentially quantifies this "chance of being wrong."

Let's calculate the Gini Impurity for a basket containing 10 fruits. We want to classify them as either Apple or Orange.

The formula is: 

$$
Gini = 1 - \sum_{i=1}^{n} (p_i)^2
$$

Gini=1−((proportion of Class 1)2+(proportion of Class 2)2+...)

1. Case 1: A Mixed Basket

    - Contents: 7 Apples and 3 Oranges.
    - Proportion of Apples (pApple​): 7/10 = 0.7
    - Proportion of Oranges (pOrange​): 3/10 = 0.3

Now, plug these into the formula:

- Gini=1−((0.7)2+(0.3)2)
- Gini=1−(0.49+0.09)
- Gini=1−0.58
- Gini Impurity = 0.42

A score of 0.42 is quite high, indicating that the basket is very mixed.


**Comparison Between Entropy & Gini Impurity**

Entropy and Gini Impurity are both metrics used by classification decision trees to measure the impurity or "mixed-up-ness" of a data node.1 The goal is always to find a feature split that results in the greatest reduction in this impurity

While they serve the same purpose and often produce very similar trees, they differ in their calculation and computational cost.3

| Feature | Gini Impurity | Entropy (used with Information Gain) |
| :--- | :--- | :--- |
| **Core Concept** | Measures the probability of **misclassifying** a randomly chosen element from the set. | Measures the **uncertainty** or **randomness** in the set, based on Information Theory. |
| **Formula (Two-Class)** | `$1 - (p_1^2 + p_2^2)$` | `$-p_1\log_2(p_1) - p_2\log_2(p_2)$` |
| **Range (Two-Class)** | `0` (pure) to `0.5` (maximally impure) | `0` (pure) to `1.0` (maximally impure) |
| **Computational Cost**| **Faster**. It only requires squaring operations. | **Slower**. It requires logarithmic calculations, which are more computationally expensive. |
| **Behavior / Curve** | An inverted parabola. | A slightly steeper, inverted arch. More sensitive to changes in probability for highly imbalanced classes. |
| **Effect on Tree** | Tends to isolate the most frequent class in its own branch. | Has a slight tendency to produce more balanced trees. |

**Difference**

While the scales are different, the shapes of their curves for a two-class problem are very similar. Both peak when the classes are perfectly balanced (50/50).

The key difference seen in the graph is that Entropy's value climbs slightly more steeply. This makes it more sensitive to small probabilities, but the overall shape and the location of the peak are the same.

**Which One to Choose 🤔**

1. Similarity in Practice: In most real-world applications, the choice between Gini Impurity and Entropy makes very little difference to the final accuracy of the tree.5 They will often choose the exact same splits.
2. Speed is a Factor: Because Gini Impurity is computationally faster, it is the default choice in many popular machine learning libraries, including scikit-learn.6
3. Minor Nuances: Entropy is slightly more sensitive to nodes that have a mix of classes.7 This can lead it to favor creating more balanced splits, but this effect is usually minor.

Conclusion: For most purposes, sticking with the default Gini Impurity is a safe and effective choice. The performance difference is typically negligible, and you benefit from the faster computation.8 Fine-tuning other hyperparameters like max_depth or min_samples_split will have a much more significant impact on your model's performance.


### Regression Split Criteria

The primary split criterion for a regression decision tree is Variance Reduction, which is most commonly measured using Mean Squared Error (MSE).

The goal is to find the feature and split point that create the most homogeneous subgroups, meaning the resulting groups have numerical values that are as close to each other as possible.

**Minimize Variance 🏡**

Unlike a classification tree that wants to create "pure" groups of a single category, a regression tree wants to create groups with a small spread of numerical values.

- High Variance (Bad): Imagine a group of houses with prices all over the map: $150k, $200k, $500k, $900k. The values are very spread out. If you use the average price to make a prediction for this group, it will be a poor estimate for most of the houses.
- Low Variance (Good): Now imagine a group of houses with prices of $310k, $320k, $325k, and $330k. The values are tightly clustered. The average price is a very accurate and reliable prediction for any house in this group.

A regression tree's split criterion is designed to move from a high-variance group (the parent node) to low-variance subgroups (the child nodes).

**Variance Reduction (MSE)**

The tree systematically finds the best split by following this process:

1. Calculate the MSE of the Parent Node: The algorithm first calculates the Mean Squared Error (MSE) for the current group of data points. MSE is the average of the squared differences between each data point's value and the group's average value. A high MSE means high variance.
2. Test All Possible Splits: The algorithm then iterates through every feature and every possible split point. For each potential split, it temporarily divides the data into two child nodes.
3. Calculate the Weighted Average MSE of the Children: For each potential split, it calculates the MSE of the resulting child nodes and then computes their weighted average.

    Weighted Avg. MSE = (size of child 1 / size of parent) * MSE(child 1) + (size of child 2 / size of parent) * MSE(child 2)

4. Calculate the Variance Reduction: The "goodness" of the split is the amount by which the variance was reduced.

    Variance Reduction = MSE(parent) - Weighted Avg. MSE(children)

5. Select the Best Split: The tree chooses the single feature and split point that result in the highest Variance Reduction. This becomes the rule for that node.


**Mean Absolute Error (MAE)**

While MSE is the most common criterion, Mean Absolute Error (MAE) can also be used.

- MAE calculates the average of the absolute differences between each point and the group's average.
- The main difference is that MAE is less sensitive to outliers than MSE. Because MSE squares the errors, large errors (from outliers) are weighted much more heavily.
- In practice, MSE is used far more often because its mathematical properties make it easier to optimize.


### Quiz --> [Quiz](./Quiz/DecisionTreeQuiz.md)

### Previous Topic --> [Non-Parametric Technique](./Non-ParametricTechnique.md)
### Next Topic --> [Neural Networks](./NeuralNetwork.md)
</div>