## Decision Tree: 50 MCQ Questions

**1. What is the primary goal of a Decision Tree?**
- [ ] A) To create a "black box" model.
- [ ] B) To predict an outcome by learning simple decision rules from data.
- [ ] C) To only handle numerical data.
- [ ] D) To create a model that is as complex as possible.

**Correct Answer: B**

**Explanation:**
- **A) To create a "black box" model.** This is incorrect. Decision trees are known as "white box" models because their decision-making process is transparent and easy to interpret.
- **B) To predict an outcome by learning simple decision rules from data.** This is correct. A decision tree builds a flowchart of "if-then-else" rules to make predictions.
- **C) To only handle numerical data.** This is incorrect. Decision trees can handle both numerical and categorical data without requiring extensive preprocessing.
- **D) To create a model that is as complex as possible.** This is incorrect. Overly complex trees are prone to overfitting. The goal is to create a simple, generalizable model.

---

**2. In a Decision Tree, what is a "leaf node"?**
- [ ] A) The process of dividing a node.
- [ ] B) A node that splits into further sub-nodes.
- [ ] C) A final outcome or decision.
- [ ] D) The starting point of the tree.

**Correct Answer: C**

**Explanation:**
- **A) The process of dividing a node.** This describes **splitting**.
- **B) A node that splits into further sub-nodes.** This describes a **decision node** or **internal node**.
- **C) A final outcome or decision.** This is correct. A leaf node (or terminal node) represents a class label or a continuous value and does not split any further.
- **D) The starting point of the tree.** This describes the **root node**.

---

**3. What is the biggest weakness of Decision Trees?**
- [ ] A) They can only be used for classification.
- [ ] B) They require extensive data normalization.
- [ ] C) They are prone to overfitting.
- [ ] D) They are difficult to interpret.

**Correct Answer: C**

**Explanation:**
- **A) They can only be used for classification.** This is false; they can be used for both classification and regression tasks.
- **B) They require extensive data normalization.** This is false; they do not require data scaling or normalization.
- **C) They are prone to overfitting.** This is correct. Decision trees can become overly complex and learn the noise in the training data, leading to poor performance on new data.
- **D) They are difficult to interpret.** This is false; their main strength is ease of interpretation.

---

**4. What does the "depth" of a node in a Decision Tree refer to?**
- [ ] A) The number of features used in the tree.
- [ ] B) The number of leaves in the tree.
- [ ] C) The length of the path from the root node to that specific node.
- [ ] D) The total number of nodes in the tree.

**Correct Answer: C**

**Explanation:**
- **A) The number of features used in the tree.** This is unrelated to the concept of depth.
- **B) The number of leaves in the tree.** This is a measure of the tree's complexity, but not depth.
- **C) The length of the path from the root node to that specific node.** This is the correct definition. The root is at depth 0, its children are at depth 1, and so on.
- **D) The total number of nodes in the tree.** This is the size of the tree, not the depth of a specific node.

---

**5. Setting a `max_depth` hyperparameter in a Decision Tree is a method to:**
- [ ] A) Handle imbalanced data.
- [ ] B) Ensure the tree is perfectly pure.
- [ ] C) Control overfitting by limiting the tree's growth.
- [ ] D) Increase the model's complexity.

**Correct Answer: C**

**Explanation:**
- **A) Handle imbalanced data.** While related to performance, `max_depth` doesn't directly address data imbalance. Other techniques like class weighting are used for that.
- **B) Ensure the tree is perfectly pure.** A limited depth might mean stopping before nodes are pure.
- **C) Control overfitting by limiting the tree's growth.** This is correct. `max_depth` is a key hyperparameter to prevent the tree from becoming too deep and memorizing the training data.
- **D) Increase the model's complexity.** Setting a `max_depth` *limits* complexity, it doesn't increase it.

---

**6. A Decision Tree that predicts a continuous numerical value (e.g., price of a house) is called a:**
- [ ] A) Greedy Tree
- [ ] B) Regression Tree
- [ ] C) Purity Tree
- [ ] D) Classification Tree

**Correct Answer: B**

**Explanation:**
- **A) Greedy Tree.** This describes the *process* of building a tree (greedy search), not its type based on the output.
- **B) Regression Tree.** This is correct. Regression trees are used when the target variable is continuous.
- **C) Purity Tree.** This is not a standard term. Purity is a concept used within trees, not a type of tree.
- **D) Classification Tree.** This type of tree predicts a discrete category (e.g., "Spam" or "Not Spam").

---

**7. The process of building a Decision Tree is described as a "greedy recursive process." What does the "greedy" part mean?**
- [ ] A) The algorithm continues splitting until every node is 100% pure.
- [ ] B) The algorithm makes the choice that seems best at the current step without planning ahead.
- [ ] C) The algorithm uses as much memory as possible.
- [ ] D) The algorithm plans several steps ahead to find the globally optimal tree.

**Correct Answer: B**

**Explanation:**
- **A) The algorithm continues splitting until every node is 100% pure.** While a greedy process *can* lead to this, the "greedy" part specifically refers to the step-by-step decision making, not the stopping criterion.
- **B) The algorithm makes the choice that seems best at the current step without planning ahead.** This is the correct definition of a greedy algorithm. It chooses the split that provides the greatest immediate information gain or purity increase.
- **C) The algorithm uses as much memory as possible.** This is unrelated to the "greedy" concept in this context.
- **D) The algorithm plans several steps ahead...** This is the opposite of a greedy approach.

---

**8. Which of the following is a measure of impurity or uncertainty in a group of data?**
- [ ] A) Pruning
- [ ] B) Variance Reduction
- [ ] C) Entropy
- [ ] D) Information Gain

**Correct Answer: C**

**Explanation:**
- **A) Pruning.** This is a technique to reduce overfitting, not a measure of impurity.
- **B) Variance Reduction.** This is the primary split criterion for regression trees, not a general measure of impurity for classification.
- **C) Entropy.** This is correct. Entropy is a key measure of randomness or impurity in a dataset, with 0 being pure and 1 being maximally impure (for a two-class problem).
- **D) Information Gain.** This measures the *reduction* in impurity after a split, not the impurity itself.

---

**9. A node has a Gini Impurity of 0. What does this signify?**
- [ ] A) The node requires further splitting.
- [ ] B) The node is the root node.
- [ ] C) The node is perfectly pure (all items belong to one class).
- [ ] D) The node is perfectly mixed (50/50 split).

**Correct Answer: C**

**Explanation:**
- **A) The node requires further splitting.** A pure node is a leaf node and does not need to be split further.
- **B) The node is the root node.** The root node is typically impure, not pure.
- **C) The node is perfectly pure (all items belong to one class).** This is correct. A Gini score of 0 means there is no impurity.
- **D) The node is perfectly mixed (50/50 split).** This would result in the maximum Gini Impurity (0.5 for a two-class problem).

---

**10. How does a Regression Tree make a prediction for a new data point?**
- [ ] A) It predicts the median of all target values in the dataset.
- [ ] B) It predicts the average of the target values of the training data in the leaf node.
- [ ] C) It predicts the value of the closest training data point.
- [ ] D) It predicts the majority class of the training data in the leaf node.

**Correct Answer: B**

**Explanation:**
- **A) It predicts the median of all target values in the dataset.** This would be a very naive model that ignores all features.
- **B) It predicts the average of the target values of the training data in the leaf node.** This is correct. The prediction of a regression tree is the mean of the outcomes for all training instances that fall into that leaf.
- **C) It predicts the value of the closest training data point.** This describes a k-Nearest Neighbors (k-NN) approach.
- **D) It predicts the majority class...** This is how a **classification tree** makes a prediction.

---

**11. What is the primary split criterion used in a Regression Tree?**
- [ ] A) Chi-Square
- [ ] B) Gini Impurity
- [ ] C) Variance Reduction (e.g., using MSE)
- [ ] D) Information Gain

**Correct Answer: C**

**Explanation:**
- **A) Chi-Square.** Another metric sometimes used for classification, but Variance Reduction is standard for regression.
- **B) Gini Impurity.** Used for classification trees.
- **C) Variance Reduction (e.g., using MSE).** This is correct. A regression tree seeks to create splits that minimize the variance (or Mean Squared Error) within the resulting child nodes.
- **D) Information Gain.** Used for classification trees.

---

**12. In a dataset with two classes, what is the maximum possible value for Entropy?**
- [ ] A) It has no upper limit.
- [ ] B) 0.5
- [ ] C) 1.0
- [ ] D) 0

**Correct Answer: C**

**Explanation:**
- **A) It has no upper limit.** This is incorrect; it is bounded.
- **B) 0.5.** This is the maximum value for **Gini Impurity** in a two-class problem.
- **C) 1.0.** This is correct. Entropy reaches its maximum value of 1.0 when the classes are perfectly balanced (e.g., 50% Class A, 50% Class B).
- **D) 0.** This is the minimum value, representing a perfectly pure node.

---

**13. Which statement is true when comparing Gini Impurity and Entropy?**
- [ ] A) Using Entropy always results in a more accurate tree than using Gini Impurity.
- [ ] B) Gini Impurity is more sensitive to changes in class probabilities than Entropy.
- [ ] C) Gini Impurity is the default criterion in many libraries (like scikit-learn) because it's computationally faster.
- [ ] D) Entropy is computationally faster to calculate than Gini Impurity.

**Correct Answer: C**

**Explanation:**
- **A) Using Entropy always results in a more accurate tree...** This is false. In practice, the difference in performance is usually negligible.
- **B) Gini Impurity is more sensitive...** This is false. Entropy is slightly more sensitive to changes, especially for imbalanced classes.
- **C) Gini Impurity is the default criterion in many libraries (like scikit-learn) because it's computationally faster.** This is correct. Due to its computational efficiency and similar performance, Gini Impurity is often the default.
- **D) Entropy is computationally faster...** This is false. The logarithm calculation in Entropy makes it slower than the squaring operation in Gini Impurity.

---

**14. The technique of removing branches from a decision tree to prevent overfitting is called:**
- [ ] A) Rooting
- [ ] B) Growing
- [ ] C) Pruning
- [ ] D) Splitting

**Correct Answer: C**

**Explanation:**
- **A) Rooting.** This refers to establishing the first node of the tree.
- **B) Growing.** This is the overall process of building the tree.
- **C) Pruning.** This is the correct term for removing sections of the tree that provide little predictive power, which helps to reduce complexity and prevent overfitting.
- **D) Splitting.** This is the process of creating branches, not removing them.

---

**15. A decision tree is built to decide if a loan should be approved. The root node contains 1000 applications (600 approved, 400 denied). This node has:**
- [ ] A) A Gini Impurity of 0.5
- [ ] B) High purity
- [ ] C) Zero impurity
- [ ] D) Low purity

**Correct Answer: D**

**Explanation:**
- **A) A Gini Impurity of 0.5.** This would only be true if the split was exactly 50/50 (500 approved, 500 denied). The Gini score here would be `1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48`, which is high, but not the maximum. "Low purity" is the best description.
- **B) High purity.** This is incorrect. A pure node would have nearly all applications belonging to one class.
- **C) Zero impurity.** This would mean all 1000 applications were either approved or denied.
- **D) Low purity.** This is correct. The node is quite mixed, with a 60/40 split. This means it has high impurity and low purity.

---

**16. Which of these is an advantage of Decision Trees?**
- [ ] A) They are guaranteed to find the globally optimal model.
- [ ] B) They are immune to the effects of imbalanced data.
- [ ] C) They require little data preparation (no need for normalization).
- [ ] D) They are very stable; small changes in data do not affect the tree.

**Correct Answer: C**

**Explanation:**
- **A) They are guaranteed to find the globally optimal model.** This is false. The greedy approach means they find a locally optimal solution, which may not be the globally best one.
- **B) They are immune to the effects of imbalanced data.** This is false. They can be biased towards the majority class in an imbalanced dataset.
- **C) They require little data preparation (no need for normalization).** This is a key advantage. The tree's splitting logic is not affected by the scale of the features.
- **D) They are very stable...** This is false. Decision trees are known to be unstable; small data changes can lead to a completely different tree.

---

**17. Information Gain is calculated as:**
- [ ] A) Entropy(parent) / Weighted Average Entropy(children)
- [ ] B) Entropy(parent) - Weighted Average Entropy(children)
- [ ] C) Weighted Average Entropy(children) - Entropy(parent)
- [ ] D) Entropy(parent) + Weighted Average Entropy(children)

**Correct Answer: B**

**Explanation:**
- **A, C, D.** These formulas are incorrect.
- **B) Entropy(parent) - Weighted Average Entropy(children).** This is the correct formula. It measures the reduction in entropy (uncertainty) achieved by the split. A higher information gain is better.

---

**18. If a decision tree is allowed to grow to its maximum possible depth with no restrictions, it is likely to result in:**
- [ ] A) High bias and low variance
- [ ] B) Overfitting
- [ ] C) A very simple model
- [ ] D) Underfitting

**Correct Answer: B**

**Explanation:**
- **A) High bias and low variance.** This describes an underfit model. An overfit model has low bias and high variance.
- **B) Overfitting.** This is correct. A deep, unrestricted tree will have low bias but high variance, meaning it has learned the training data too perfectly, including its noise, and will not generalize well.
- **C) A very simple model.** An unrestricted tree is typically very complex.
- **D) Underfitting.** This occurs when a model is too simple (e.g., a very shallow tree).

---

**19. In a classification tree, the algorithm splits a node based on the feature that:**
- [ ] A) Is a numerical feature.
- [ ] B) Results in the greatest reduction in impurity.
- [ ] C) Has the most unique values.
- [ ] D) Results in the highest increase in impurity.

**Correct Answer: B**

**Explanation:**
- **A) Is a numerical feature.** The best split can come from either numerical or categorical features.
- **B) Results in the greatest reduction in impurity.** This is correct. The algorithm chooses the split that yields the "purest" child nodes, which is measured as the highest Information Gain (reduction in Entropy) or the greatest reduction in Gini Impurity.
- **C) Has the most unique values.** This is not a criterion for a good split and can sometimes be misleading.
- **D) Results in the highest increase in impurity.** The goal is to decrease impurity, not increase it.

---

**20. A "white box" model means that:**
- [ ] A) The model can only be trained on text data.
- [ ] B) The model's internal logic is hidden and difficult to understand.
- [ ] C) The model's decision-making process is transparent and interpretable.
- [ ] D) The model is a neural network.

**Correct Answer: C**

**Explanation:**
- **A) The model can only be trained on text data.** This is unrelated to the "white box" concept.
- **B) The model's internal logic is hidden...** This describes a "black box" model.
- **C) The model's decision-making process is transparent and interpretable.** This is the correct definition of a "white box" model, and Decision Trees are a prime example.
- **D) The model is a neural network.** Neural networks are typically considered "black box" models.

---

**21. A decision tree is considered "non-parametric" because:**
- [ ] A) It is not a machine learning model.
- [ ] B) It does not make strong assumptions about the underlying distribution of the data.
- [ ] C) It can only be used for non-linear data.
- [ ] D) It has no parameters to tune.

**Correct Answer: B**

**Explanation:**
- **A) It is not a machine learning model.** This is false.
- **B) It does not make strong assumptions about the underlying distribution of the data.** This is correct. Unlike linear regression, which assumes a linear relationship, decision trees build rules based on the data they observe without assuming a specific functional form.
- **C) It can only be used for non-linear data.** This is false. It can model both linear and non-linear relationships.
- **D) It has no parameters to tune.** This is false. It has hyperparameters like `max_depth` and `min_samples_leaf`.

---

**22. What is the role of the `min_samples_split` hyperparameter?**
- [ ] A) It determines the impurity measure to use.
- [ ] B) It sets the maximum depth of the tree.
- [ ] C) It sets the minimum number of samples required to split an internal node.
- [ ] D) It sets the minimum number of samples required to be at a leaf node.

**Correct Answer: C**

**Explanation:**
- **A) It determines the impurity measure to use.** This is determined by the `criterion` hyperparameter (e.g., 'gini' or 'entropy').
- **B) It sets the maximum depth of the tree.** This describes the `max_depth` hyperparameter.
- **C) It sets the minimum number of samples required to split an internal node.** This is correct. It is a pre-pruning technique to control overfitting by preventing splits on nodes with too few samples.
- **D) It sets the minimum number of samples required to be at a leaf node.** This describes the `min_samples_leaf` hyperparameter.

---

**23. If a split on a feature results in an Information Gain of 0, what does this imply?**
- [ ] A) The feature is a categorical feature.
- [ ] B) The split resulted in perfectly pure child nodes.
- [ ] C) The split provided no new information about the classification.
- [ ] D) The split is the best possible split.

**Correct Answer: C**

**Explanation:**
- **A) The feature is a categorical feature.** The type of feature is irrelevant to the information gain value.
- **B) The split resulted in perfectly pure child nodes.** This would likely result in a high information gain, not zero.
- **C) The split provided no new information about the classification.** This is correct. An Information Gain of 0 means the distribution of classes in the child nodes is the same as in the parent node, so the split was not useful.
- **D) The split is the best possible split.** This is incorrect. An Information Gain of 0 means it's a useless split.

---

**24. In a regression tree, the impurity of a node is measured by:**
- [ ] A) The number of samples in the node
- [ ] B) Entropy
- [ ] C) Variance or Mean Squared Error (MSE)
- [ ] D) Gini Impurity

**Correct Answer: C**

**Explanation:**
- **A) The number of samples in the node.** This is a property of the node, not a measure of its impurity.
- **B) Entropy.** Used for classification.
- **C) Variance or Mean Squared Error (MSE).** This is correct. Regression trees aim to create groups where the numerical target values are as close as possible, meaning they have low variance.
- **D) Gini Impurity.** Used for classification.

---

**25. A small change in the training data causes a large change in the decision tree's structure. This property is known as:**
- [ ] A) Bias
- [ ] B) Robustness
- [ ] C) Instability
- [ ] D) Stability

**Correct Answer: C**

**Explanation:**
- **A) Bias.** Bias refers to the simplifying assumptions made by a model, not its reaction to data changes.
- **B) Robustness.** A robust model is not easily affected by small changes or outliers.
- **C) Instability.** This is correct. Decision trees are known to be unstable because the greedy splitting process can be significantly altered by minor variations in the data.
- **D) Stability.** This is the opposite of the described behavior.

---

**26. How do decision trees handle missing values in data?**
- [ ] A) They treat missing values as a separate category.
- [ ] B) They cannot handle missing values and will raise an error.
- [ ] C) Some algorithms can ignore them or use surrogate splits.
- [ ] D) They automatically impute the mean/median for the missing values.

**Correct Answer: C**

**Explanation:**
- **A) They treat missing values as a separate category.** While this is a possible preprocessing strategy, it's not the only way trees handle them.
- **B) They cannot handle missing values...** Many implementations (like scikit-learn) do require imputation beforehand, but more advanced algorithms like C4.5 or CART have built-in mechanisms.
- **C) Some algorithms can ignore them or use surrogate splits.** This is correct. Advanced tree algorithms like CART can use "surrogate splits" (alternative splits) when a value is missing.
- **D) They automatically impute the mean/median...** This is a separate preprocessing step, not an inherent capability of all tree algorithms.

---

**27. What is the "bias-variance trade-off" in the context of a decision tree's `max_depth`?**
- [ ] A) A deep tree has high bias and high variance.
- [ ] B) A deep tree has low bias and low variance.
- [ ] C) A shallow tree has high bias and low variance.
- [ ] D) A shallow tree has high bias and high variance.

**Correct Answer: C**

**Explanation:**
- **A) A deep tree has high bias and high variance.** Incorrect. Low bias.
- **B) A deep tree has low bias and low variance.** Incorrect. High variance.
- **C) A shallow tree has high bias and low variance.** This is correct. A simple (shallow) tree makes strong assumptions (high bias) but is stable and doesn't change much with new data (low variance).
- **D) A shallow tree has high bias and high variance.** Incorrect. Low variance.

---

**28. The first split in a decision tree is called the:**
- [ ] A) Branch
- [ ] B) Decision Node
- [ ] C) Root Node
- [ ] D) Leaf Node

**Correct Answer: C**

**Explanation:**
- **A) Branch.** This is the connection between nodes.
- **B) Decision Node.** This is a generic term for any node that splits.
- **C) Root Node.** This is the correct term for the very first node/split in the tree, representing the entire dataset.
- **D) Leaf Node.** This is a terminal node with a final decision.

---

**29. Which of the following is a method of "pre-pruning" a decision tree?**
- [ ] A) Converting the tree into a set of rules.
- [ ] B) Setting a minimum value for information gain to make a split.
- [ ] C) Using a separate validation set to evaluate the tree.
- [ ] D) Growing the tree to its full depth and then removing branches.

**Correct Answer: B**

**Explanation:**
- **A) Converting the tree into a set of rules.** This is a way to interpret the tree, not to prune it.
- **B) Setting a minimum value for information gain to make a split.** This is correct. Pre-pruning involves setting stopping conditions *before* or *during* the tree's construction, such as setting `max_depth`, `min_samples_split`, or a threshold for impurity reduction.
- **C) Using a separate validation set to evaluate the tree.** This is part of the overall modeling process, not a pruning technique itself.
- **D) Growing the tree to its full depth and then removing branches.** This describes **post-pruning**.

---

**30. A decision tree can be used to visually represent:**
- [ ] A) A neural network architecture.
- [ ] B) A series of if-then-else rules.
- [ ] C) The distribution of a single variable.
- [ ] D) The correlation between two variables.

**Correct Answer: B**

**Explanation:**
- **A) A neural network architecture.** This is represented by a graph of neurons and layers.
- **B) A series of if-then-else rules.** This is correct. The flowchart-like structure of a decision tree is a direct visual representation of a nested set of if-then-else statements.
- **C) The distribution of a single variable.** This is shown with a histogram or density plot.
- **D) The correlation between two variables.** This is typically shown with a scatter plot.

---

**31. In a two-class problem, a node contains 8 samples of Class A and 8 samples of Class B. The Gini Impurity is:**
- [ ] A) 1.0
- [ ] B) 0.25
- [ ] C) 0.5
- [ ] D) 0

**Correct Answer: C**

**Explanation:**
- The proportions are p(A) = 8/16 = 0.5 and p(B) = 8/16 = 0.5.
- Gini = 1 - (p(A)^2 + p(B)^2) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 1 - 0.5 = 0.5.
- **C) 0.5** is correct. This represents the maximum possible Gini Impurity for a two-class problem.

---

**32. Why are ensemble methods like Random Forests often preferred over a single Decision Tree?**
- [ ] A) They require less data.
- [ ] B) They are computationally faster to train.
- [ ] C) They reduce overfitting and improve stability.
- [ ] D) They are easier to interpret.

**Correct Answer: C**

**Explanation:**
- **A) They require less data.** They typically require more data to be effective.
- **B) They are computationally faster to train.** This is false. Training hundreds of trees is slower than training one.
- **C) They reduce overfitting and improve stability.** This is correct. By averaging the predictions of many different trees (which have been trained on different subsets of data), Random Forests reduce the high variance and instability of single decision trees.
- **D) They are easier to interpret.** This is false. Ensembles of trees are much harder to interpret than a single tree.

---

**33. A split on a numerical feature (e.g., Age) in a decision tree involves:**
- [ ] A) Ignoring the feature because it is not categorical.
- [ ] B) Finding a single threshold (e.g., Age > 30) that best separates the data.
- [ ] C) Converting the Age into categories (e.g., Young, Middle-aged, Old) first.
- [ ] D) Creating a branch for every unique value of Age.

**Correct Answer: B**

**Explanation:**
- **A) Ignoring the feature...** Decision trees are well-suited for numerical features.
- **B) Finding a single threshold (e.g., Age > 30) that best separates the data.** This is correct. The algorithm tests all possible split points for the numerical feature and selects the one that maximizes the impurity reduction.
- **C) Converting the Age into categories...** This is a possible preprocessing step (binning), but the tree algorithm itself works by finding a threshold on the continuous values.
- **D) Creating a branch for every unique value...** This is inefficient and leads to overfitting. It's typically done for categorical features.

---

**34. The term "recursive partitioning" refers to:**
- [ ] A) The pruning of the decision tree.
- [ ] B) The process of repeatedly splitting the dataset into smaller and smaller subsets.
- [ ] C) The calculation of Gini Impurity.
- [ ] D) The final prediction made by the tree.

**Correct Answer: B**

**Explanation:**
- **A) The pruning of the decision tree.** This is a separate step to control complexity.
- **B) The process of repeatedly splitting the dataset into smaller and smaller subsets.** This is the correct definition. The algorithm is "recursive" because the same splitting logic is applied to each new subset (node) that is created.
- **C) The calculation of Gini Impurity.** This is a metric used within the process, not the process itself.
- **D) The final prediction made by the tree.** This is the output, not the process.

---

**35. If your decision tree model has very high accuracy on the training data but low accuracy on the test data, you are likely experiencing:**
- [ ] A) Data leakage
- [ ] B) High bias
- [ ] C) Overfitting
- [ ] D) Underfitting

**Correct Answer: C**

**Explanation:**
- **A) Data leakage.** While this can cause high performance, overfitting is the more direct description of the performance gap between training and testing.
- **B) High bias.** This is another term for underfitting.
- **C) Overfitting.** This is the classic symptom of overfitting (also called high variance). The model has learned the training data too well, including its noise, and fails to generalize to new, unseen data.
- **D) Underfitting.** This would result in low accuracy on *both* training and test data.

---

**36. Which impurity metric is generally more sensitive to changes in the probabilities of the classes?**
- [ ] A) Both Gini and Entropy are equally sensitive.
- [ ] B) Entropy
- [ ] C) Mean Squared Error
- [ ] D) Gini Impurity

**Correct Answer: B**

**Explanation:**
- **A) Both Gini and Entropy are equally sensitive.** This is incorrect; there is a slight difference in their sensitivity curves.
- **B) Entropy.** Correct. The logarithmic function in the entropy calculation makes it slightly more sensitive to changes, especially when classes are imbalanced.
- **C) Mean Squared Error.** This is for regression, not classification impurity.
- **D) Gini Impurity.** Less sensitive due to its parabolic shape.

---

**37. A decision boundary created by a decision tree is typically:**
- [ ] A) A circle
- [ ] B) A straight line
- [ ] C) A series of axis-parallel lines (a step function)
- [ ] D) A smooth curve

**Correct Answer: C**

**Explanation:**
- **A) A circle.** This is not a typical decision boundary for a tree.
- **B) A straight line.** This is characteristic of linear models like Logistic Regression or Linear SVM.
- **C) A series of axis-parallel lines (a step function).** This is correct. Because each split is made on a single feature (e.g., `X1 > 5` or `X2 < 10`), the resulting decision boundaries are always perpendicular to the feature axes, creating rectangular regions.
- **D) A smooth curve.** This is characteristic of models like Support Vector Machines with non-linear kernels.

---

**38. What happens if you set `min_samples_leaf` to a very high value?**
- [ ] A) It will have no effect on the tree's structure.
- [ ] B) The tree will be more likely to underfit.
- [ ] C) The tree will become more complex.
- [ ] D) The tree will be more likely to overfit.

**Correct Answer: B**

**Explanation:**
- **A) It will have no effect...** This is incorrect; it's a key hyperparameter.
- **B) The tree will be more likely to underfit.** This is correct. A high value for `min_samples_leaf` means that a split is only considered if the resulting leaves are very large. This prevents the tree from making fine-grained distinctions, leading to a simpler model that may underfit (high bias).
- **C) The tree will become more complex.** This is incorrect; it will become simpler.
- **D) The tree will be more likely to overfit.** This is incorrect. Overfitting is associated with low values for `min_samples_leaf`.

---

**39. The CART (Classification and Regression Trees) algorithm uses which of the following for splitting?**
- [ ] A) Information Gain for both classification and regression.
- [ ] B) Gini Impurity for classification and Variance Reduction for regression.
- [ ] C) Gini Impurity for both classification and regression.
- [ ] D) Information Gain for classification and Variance Reduction for regression.

**Correct Answer: B**

**Explanation:**
- The CART algorithm, which is one of the most common foundations for decision tree implementations (including scikit-learn), is designed to use **Gini Impurity** for classification tasks and **Variance Reduction (MSE)** for regression tasks.

---

**40. A key advantage of decision trees over linear models is that they:**
- [ ] A) Require feature scaling.
- [ ] B) Are computationally faster to train.
- [ ] C) Can capture non-linear relationships between features and the target.
- [ ] D) Are less prone to overfitting.

**Correct Answer: C**

**Explanation:**
- **A) Require feature scaling.** This is false; they do not require it.
- **B) Are computationally faster to train.** This is not always true, especially for large datasets.
- **C) Can capture non-linear relationships between features and the target.** This is a major advantage. The hierarchical splitting process allows trees to model complex, non-linear patterns that linear models cannot.
- **D) Are less prone to overfitting.** This is false; they are more prone to overfitting.

---

**41. What is a "surrogate split"?**
- [ ] A) A split based on a randomly chosen feature.
- [ ] B) An alternative split used when a data point has a missing value for the primary split feature.
- [ ] C) The final split in a branch before a leaf node.
- [ ] D) A split that results in zero information gain.

**Correct Answer: B**

**Explanation:**
- **A) A split based on a randomly chosen feature.** This is a technique used in Random Forests, not a surrogate split.
- **B) An alternative split used when a data point has a missing value for the primary split feature.** This is correct. In algorithms like CART, if the best feature to split on is missing for a particular sample, the algorithm can use a "surrogate" feature that has a similar splitting effect to route the sample down the tree.
- **C) The final split in a branch...** This is just a decision node.
- **D) A split that results in zero information gain.** This is just a poor split.

---

**42. If a feature is very important for a decision tree model, it will likely be used:**
- [ ] A) To create the final leaf nodes.
- [ ] B) Near the root of the tree.
- [ ] C) In multiple branches simultaneously.
- [ ] D) Only in the deepest parts of the tree.

**Correct Answer: B**

**Explanation:**
- **A) To create the final leaf nodes.** Splits create decision nodes, not leaf nodes.
- **B) Near the root of the tree.** This is correct. The greedy nature of the algorithm means it will select the most powerful, most informative features for the first splits (near the root) because they provide the largest impurity reduction for the entire dataset.
- **C) In multiple branches simultaneously.** A feature can be used multiple times in a tree, but its importance is indicated by its position.
- **D) Only in the deepest parts of the tree.** Less important features are used deeper in the tree.

---

**43. The process of converting a decision tree into a set of if-then rules results in rules that are:**
- [ ] A) Mutually exclusive and exhaustive.
- [ ] B) Difficult to understand.
- [ ] C) Always linear.
- [ ] D) Overlapping and complex.

**Correct Answer: A**

**Explanation:**
- **A) Mutually exclusive and exhaustive.** This is correct. Each path from the root to a leaf forms a distinct rule. Because a data point can only follow one path, the rules are mutually exclusive (it can't satisfy two rules at once). The set of all rules covers all possible data points, making them exhaustive.
- **B) Difficult to understand.** The rules are typically very easy to understand.
- **C) Always linear.** The rules are logical, not necessarily linear.
- **D) Overlapping and complex.** The rules are not overlapping.

---

**44. In a two-class problem, a node has an Entropy of 1. This means the node contains:**
- [ ] A) More samples from Class A than Class B.
- [ ] B) Only samples from Class B.
- [ ] C) An equal number of samples from Class A and Class B.
- [ ] D) Only samples from Class A.

**Correct Answer: C**

**Explanation:**
- **A) More samples from Class A than Class B.** This would result in an Entropy value between 0 and 1, but not 1.
- **B) Only samples from Class B.** This would be an Entropy of 0.
- **C) An equal number of samples from Class A and Class B.** This is correct. Entropy is maximized (at a value of 1 for a two-class problem) when there is maximum uncertainty, which occurs with a 50/50 split of the classes.
- **D) Only samples from Class A.** This would be an Entropy of 0.

---

**45. Which of the following is a disadvantage of the greedy approach used to build decision trees?**
- [ ] A) It produces trees that are too simple.
- [ ] B) It may not find the globally optimal tree.
- [ ] C) It only works for classification problems.
- [ ] D) It is computationally very expensive.

**Correct Answer: B**

**Explanation:**
- **A) It produces trees that are too simple.** It often produces trees that are too complex and overfit.
- **B) It may not find the globally optimal tree.** This is the core disadvantage. By making the best choice at each local step, it might miss a sequence of "less good" initial splits that would have led to a better overall tree.
- **C) It only works for classification problems.** It works for regression as well.
- **D) It is computationally very expensive.** The greedy approach is actually a heuristic to make the problem computationally feasible. Finding the truly optimal tree is an NP-hard problem.

---

**46. The C4.5 algorithm is an extension of which earlier algorithm?**
- [ ] A) MARS
- [ ] B) CHAID
- [ ] C) ID3
- [ ] D) CART

**Correct Answer: C**

**Explanation:**
- **A) MARS.** This is a different type of regression algorithm.
- **B) CHAID.** This is another, different tree algorithm.
- **C) ID3.** This is correct. Ross Quinlan developed the ID3 algorithm, and later improved upon it with C4.5, which added features like handling missing values and continuous attributes.
- **D) CART.** CART was developed concurrently with ID3 and has its own lineage.

---

**47. A decision tree is trained to predict if a customer will click on an ad. The feature "Country" has 50 unique values. What is a potential problem with using this feature directly?**
- [ ] A) The feature has no predictive power.
- [ ] B) The feature will cause the tree to underfit.
- [ ] C) The tree might create a large number of branches, leading to overfitting.
- [ ] D) The tree cannot handle categorical features.

**Correct Answer: C**

**Explanation:**
- **A) The feature has no predictive power.** We cannot know this without testing it, but the high cardinality itself is a structural problem.
- **B) The feature will cause the tree to underfit.** It's more likely to cause overfitting.
- **C) The tree might create a large number of branches, leading to overfitting.** This is correct. High-cardinality categorical features can cause the tree to create many small, specific splits, which may not generalize well to new data. The model starts to memorize the data instead of learning patterns.
- **D) The tree cannot handle categorical features.** This is false.

---

**48. Post-pruning a decision tree involves:**
- [ ] A) Stopping the tree growth early.
- [ ] B) Growing the tree fully, then removing or collapsing nodes.
- [ ] C) Using Gini Impurity instead of Entropy.
- [ ] D) Setting `max_depth` before training.

**Correct Answer: B**

**Explanation:**
- **A) Stopping the tree growth early.** This is pre-pruning.
- **B) Growing the tree fully, then removing or collapsing nodes.** This is the correct definition of post-pruning (or just "pruning"). It allows the tree to grow to a complex state and then simplifies it by removing nodes that do not significantly improve its performance on a validation set.
- **C) Using Gini Impurity instead of Entropy.** This is a choice of splitting criterion, not a pruning method.
- **D) Setting `max_depth` before training.** This is pre-pruning.

---

**49. In a regression tree, if a leaf node contains the values [10, 20, 30, 40], what would the tree's prediction be for a new sample that lands in this leaf?**
- [ ] A) 40
- [ ] B) 25
- [ ] C) 30
- [ ] D) 10

**Correct Answer: B**

**Explanation:**
- The prediction for a leaf in a regression tree is the average of the target values of the training samples in that leaf.
- Average = (10 + 20 + 30 + 40) / 4 = 100 / 4 = 25.
- **B) 25** is the correct answer.

---

**50. Decision trees are fundamental components of which popular ensemble algorithm?**
- [ ] A) Random Forest
- [ ] B) Logistic Regression
- [ ] C) Support Vector Machines
- [ ] D) k-Nearest Neighbors

**Correct Answer: A**

**Explanation:**
- **A) Random Forest.** This is correct. A Random Forest is an ensemble model that builds and aggregates the results from a multitude of individual decision trees to produce a more robust and accurate prediction.
- **B, C, D.** These are all distinct machine learning algorithms.