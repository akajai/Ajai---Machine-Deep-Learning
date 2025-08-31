
**1. In a scenario with highly overlapping class distributions, a Bayesian classifier with a zero-one loss function will have a Bayes error rate that is:**
- [ ] A) Exactly zero, as the classifier is optimal.
- [ ] B) Dependent on the number of features, but not the overlap.
- [ ] C) Greater than zero, representing the irreducible error due to class overlap.
- [ ] D) Equal to the highest prior probability among the classes.

**Correct Answer:** C

**Explanation:**
- A) The Bayes error is the minimum possible error, but it is only zero if the classes are perfectly separable.
- B) The number of features affects the complexity, but the overlap in distributions is the direct cause of irreducible error.
- C) The Bayes error rate is the minimum achievable error, which is non-zero when the class-conditional densities p(x|ω_i) overlap. This overlap means some data points could plausibly belong to multiple classes, making some error unavoidable.
- D) The error is related to the posterior probabilities, not just the highest prior.

**2. A key difference between a discriminative model (e.g., Logistic Regression) and a generative model (e.g., Naive Bayes) is that:**
- [ ] A) A generative model learns the decision boundary directly, while a discriminative model learns the underlying data distribution.
- [ ] B) A discriminative model learns the posterior probability P(ω|x) directly, while a generative model learns the prior P(ω) and likelihood p(x|ω) to compute the posterior.
- [ ] C) Generative models are only suitable for regression, while discriminative models are for classification.
- [ ] D) Discriminative models are always linear, whereas generative models can be non-linear.

**Correct Answer:** B

**Explanation:**
- A) This statement reverses the definitions.
- B) This is the fundamental distinction. Discriminative models focus on finding the boundary between classes. Generative models learn the distribution of each class and use that to make a decision, which allows them to also generate new data samples.
- C) Both model types are primarily used for classification.
- D) Both model types can be linear or non-linear.

**3. In Principal Component Analysis (PCA), the first principal component (PC1) is the eigenvector of the covariance matrix that corresponds to:**
- [ ] A) The smallest eigenvalue, representing the direction of minimum variance.
- [ ] B) The largest eigenvalue, representing the direction of maximum variance.
- [ ] C) An eigenvalue of zero, representing a redundant feature.
- [ ] D) An eigenvalue equal to the number of dimensions in the data.

**Correct Answer:** B

**Explanation:**
- A) The direction of minimum variance is captured by the last principal component.
- B) PCA is designed to find the orthogonal axes that capture the most variance in the data. The eigenvector associated with the largest eigenvalue points in the direction of the greatest data spread.
- C) An eigenvalue of zero would imply no variance in that direction.
- D) The sum of the eigenvalues equals the total variance, not a single eigenvalue.

**4. The "kernel trick" in Support Vector Machines (SVMs) allows the algorithm to:**
- [ ] A) Operate in a high-dimensional feature space without explicitly computing the coordinates of the data in that space.
- [ ] B) Reduce the number of support vectors required for the decision boundary.
- [ ] C) Guarantee that the data will be linearly separable in the original feature space.
- [ ] D) Select the optimal regularization parameter C automatically.

**Correct Answer:** A

**Explanation:**
- A) This is the essence of the kernel trick. It uses a kernel function to compute the dot product of data points in a higher-dimensional space, which is computationally much cheaper than performing the explicit transformation.
- B) The number of support vectors is not directly controlled by the kernel trick.
- C) The trick is used precisely because the data is *not* linearly separable in the original space.
- D) The choice of C is a separate hyperparameter tuning problem.

**5. In the context of the bias-variance tradeoff, a Random Forest model with a large number of trees typically has:**
- [ ] A) High bias and high variance.
- [ ] B) Low bias and high variance.
- [ ] C) High bias and low variance.
- [ ] D) Low bias and low variance.

**Correct Answer:** D

**Explanation:**
- A) This describes a poor model.
- B) This describes a single, deep, unpruned decision tree that is overfitting.
- C) This describes a simple, underfitting model.
- D) Individual trees in a Random Forest have low bias and high variance. By averaging the predictions of many decorrelated trees, the variance is significantly reduced while the low bias is maintained, resulting in a model with both low bias and low variance.

**6. The primary motivation for using a non-parametric density estimation technique like Kernel Density Estimation (KDE) over a parametric one is:**
- [ ] A) Non-parametric methods are computationally faster and require less data.
- [ ] B) Non-parametric methods do not make strong assumptions about the underlying shape of the data distribution.
- [ ] C) Parametric methods can only be used for univariate data.
- [ ] D) Non-parametric methods always produce a model with lower bias.

**Correct Answer:** B

**Explanation:**
- A) Non-parametric methods are generally more computationally intensive and require more data.
- B) This is the key advantage. If the data does not fit a standard distribution (like Gaussian), a parametric method will produce a poor model. KDE is flexible and lets the data define the shape of the density curve.
- C) Parametric methods can be applied to multivariate data (e.g., multivariate Gaussian).
- D) While flexible (low bias), a poor choice of bandwidth in KDE can lead to a model with high variance.

**7. In Gradient Boosting, each new weak learner is trained to predict:**
- [ ] A) The class labels of the training data directly.
- [ ] B) The residual errors of the current ensemble's predictions.
- [ ] C) The weights of the misclassified instances from the previous learner.
- [ ] D) A random subset of the features.

**Correct Answer:** B

**Explanation:**
- A) This is how a standard classifier is trained, not a boosting component.
- B) This is the core mechanism of Gradient Boosting. Each new model is trained to correct the mistakes (residuals) of the ensemble of models that came before it.
- C) This describes the mechanism of AdaBoost, not Gradient Boosting.
- D) This is a technique used in Random Forest.

**8. A key difference between the k-Nearest Neighbors (k-NN) algorithm and the Parzen window (KDE) method for density estimation is:**
- [ ] A) k-NN is a supervised method, while the Parzen window is unsupervised.
- [ ] B) The Parzen window uses a fixed volume (defined by bandwidth h) and counts the variable number of points inside, while k-NN uses a fixed number of points (k) and measures the variable volume they occupy.
- [ ] C) k-NN can only be used for classification, while the Parzen window is only for density estimation.
- [ ] D) The Parzen window is a parametric method, while k-NN is non-parametric.

**Correct Answer:** B

**Explanation:**
- A) Both are non-parametric density estimation techniques, which are unsupervised.
- B) This is the fundamental distinction. KDE fixes the window size (volume) and k varies. k-NN fixes k and the volume varies.
- C) Both can be adapted for classification and density estimation.
- D) Both are non-parametric methods.

**9. The decision boundary of a Bayes classifier for two classes with identical covariance matrices but different means is:**
- [ ] A) A quadratic curve.
- [ ] B) A linear hyperplane.
- [ ] C) A sphere centered at the origin.
- [ ] D) Two parallel hyperplanes.

**Correct Answer:** B

**Explanation:**
- A) A quadratic boundary arises when the covariance matrices are different.
- B) When the covariance matrices are identical, the quadratic terms in the discriminant function cancel out, leaving a linear decision boundary.
- C) This is not a typical decision boundary shape.
- D) The decision boundary is a single surface of separation.

**10. The primary purpose of the `min_samples_leaf` hyperparameter in a decision tree is to:**
- [ ] A) Ensure that the tree grows to its maximum possible depth.
- [ ] B) Control overfitting by preventing the creation of leaf nodes that contain very few samples.
- [ ] C) Determine the impurity metric used for splitting (Gini or Entropy).
- [ ] D) Increase the model's bias by simplifying the splits.

**Correct Answer:** B

**Explanation:**
- A) This would be achieved by not setting any stopping criteria.
- B) This is a pre-pruning technique. By requiring a minimum number of samples to be present in a leaf, it prevents the tree from creating highly specific rules that only apply to a few instances, thus controlling complexity and reducing overfitting.
- C) This is controlled by the `criterion` hyperparameter.
- D) While it does simplify the tree, its direct purpose is to control overfitting by managing leaf size.

**11. In logistic regression, the coefficients (β) represent the change in the:**
- [ ] A) Probability of the outcome for a one-unit change in the predictor variable.
- [ ] B) Odds of the outcome for a one-unit change in the predictor variable.
- [ ] C) Log-odds of the outcome for a one-unit change in the predictor variable.
- [ ] D) Likelihood of the outcome for a one-unit change in the predictor variable.

**Correct Answer:** C

**Explanation:**
- A) The relationship with probability is non-linear due to the sigmoid function.
- B) The exponent of the coefficient, exp(β), represents the change in the odds.
- C) The logistic regression model is a linear model for the log-odds. Therefore, the coefficients directly represent the change in the log-odds for a one-unit change in the corresponding feature.
- D) Likelihood is a different statistical concept.

**12. The "curse of dimensionality" primarily refers to the phenomenon where:**
- [ ] A) The time required to train a model decreases as the number of features increases.
- [ ] B) The feature space becomes increasingly sparse as dimensions are added, making it difficult to find meaningful patterns.
- [ ] C) Linear models become more effective than non-linear models in high-dimensional spaces.
- [ ] D) The number of principal components in PCA must equal the number of original dimensions.

**Correct Answer:** B

**Explanation:**
- A) Training time almost always increases with more features.
- B) As dimensions increase, the volume of the space grows exponentially. To maintain the same data density, an exponentially larger amount of data is needed. With a fixed amount of data, the space becomes sparse, and the concept of "nearest neighbor" becomes less meaningful.
- C) This is not necessarily true.
- D) The goal of PCA is to use fewer components than dimensions.

**13. The primary difference between L1 (Lasso) and L2 (Ridge) regularization is that:**
- [ ] A) L1 regularization can shrink some feature coefficients to exactly zero, effectively performing feature selection, while L2 only shrinks them towards zero.
- [ ] B) L2 regularization is computationally faster than L1 regularization.
- [ ] C) L1 regularization is used for classification, while L2 regularization is used for regression.
- [ ] D) L2 regularization is more effective at handling correlated features than L1 regularization.

**Correct Answer:** A

**Explanation:**
- A) The L1 penalty term (sum of absolute values of coefficients) has the effect of forcing some coefficients to become exactly zero, which is useful for feature selection. The L2 penalty (sum of squared coefficients) does not have this property.
- B) This is not necessarily true.
- C) Both can be used for both regression and classification.
- D) Ridge (L2) is generally preferred when features are highly correlated.

**14. In an RBF network, the K-Means algorithm is used in an unsupervised manner to:**
- [ ] A) Determine the weights of the output layer.
- [ ] B) Classify the data points directly.
- [ ] C) Find the optimal centers (μ) for the Gaussian functions in the hidden layer.
- [ ] D) Select the optimal number of hidden neurons (K).

**Correct Answer:** C

**Explanation:**
- A) The output layer is trained in a supervised manner.
- B) K-Means is a clustering algorithm, not a classification algorithm for the final output.
- C) This is the core of the hybrid training process. K-Means finds the natural cluster centers in the input data, which are then used as the centers for the RBF neurons.
- D) The number of clusters (K) is a hyperparameter that must be specified beforehand.

**15. The Perceptron Convergence Theorem guarantees that the algorithm will find a separating hyperplane if:**
- [ ] A) The data is non-linearly separable.
- [ ] B) The learning rate is sufficiently small.
- [ ] C) The data is linearly separable.
- [ ] D) The number of features is less than the number of data points.

**Correct Answer:** C

**Explanation:**
- A) The theorem does not apply to non-linearly separable data.
- B) The learning rate affects the number of steps but not the guarantee of convergence itself.
- C) This is the fundamental condition for the theorem. If a linear separator exists, the algorithm is guaranteed to find one in a finite number of steps.
- D) This is not a condition for the theorem.

**16. The Adam optimizer combines the key ideas of which two other optimization algorithms?**
- [ ] A) Stochastic Gradient Descent (SGD) and Newton's Method.
- [ ] B) Momentum and RMSProp.
- [ ] C) Adagrad and L-BFGS.
- [ ] D) Batch Gradient Descent and Conjugate Gradient.

**Correct Answer:** B

**Explanation:**
- B) Adam (Adaptive Moment Estimation) computes adaptive learning rates for each parameter by using estimates of both the first moment (the mean, as in Momentum) and the second moment (the uncentered variance, as in RMSProp) of the gradients.

**17. A major advantage of Decision Trees over linear models like Logistic Regression is their ability to:**
- [ ] A) Provide confidence scores for their predictions.
- [ ] B) Handle very large datasets efficiently.
- [ ] C) Capture complex non-linear relationships and interactions between features automatically.
- [ ] D) Avoid overfitting without the need for regularization.

**Correct Answer:** C

**Explanation:**
- A) Logistic regression naturally provides probabilities.
- B) Linear models are often faster to train on very large datasets.
- C) The hierarchical, recursive partitioning nature of decision trees allows them to model complex, non-linear decision boundaries and feature interactions without requiring manual feature engineering (like creating polynomial terms).
- D) Decision trees are highly prone to overfitting and require regularization (pruning).

**18. The "naïve" assumption in Naive Bayes, while often violated in practice, is beneficial because it:**
- [ ] A) Allows the model to capture complex dependencies between features.
- [ ] B) Guarantees that the model will achieve the Bayes optimal error rate.
- [ ] C) Drastically simplifies the computation of the likelihood, making the algorithm very efficient.
- [ ] D) Ensures that the posterior probabilities are always well-calibrated.

**Correct Answer:** C

**Explanation:**
- A) It does the opposite; it assumes no dependencies.
- B) This is not guaranteed.
- C) By assuming conditional independence, the joint likelihood P(x|ω) can be calculated as the product of the individual likelihoods P(x_i|ω), which is computationally very fast and requires less data to estimate.
- D) The probabilities can be poorly calibrated, even if the final classification is correct.

**19. In AdaBoost, the "amount of say" (alpha) for a weak learner is calculated based on its:**
- [ ] A) Training time.
- [ ] B) Total error rate on the weighted training set.
- [ ] C) Number of features used.
- [ ] D) Correlation with the previous weak learner.

**Correct Answer:** B

**Explanation:**
- B) The weight of a weak learner in the final ensemble is a function of its error. A learner with a lower error rate gets a higher "amount of say," giving it more influence on the final prediction.

**20. The primary purpose of a validation set in the model development cycle is to:**
- [ ] A) Provide an unbiased estimate of the final model's performance on unseen data.
- [ ] B) Train the model's parameters.
- [ ] C) Tune the model's hyperparameters and prevent overfitting.
- [ ] D) Augment the training data to increase its size.

**Correct Answer:** C

**Explanation:**
- A) This is the purpose of the test set.
- B) This is the purpose of the training set.
- C) The validation set is used to evaluate the model's performance on data it wasn't trained on during the development process. This feedback is used to tune hyperparameters (like learning rate, tree depth, etc.) and for techniques like early stopping to prevent overfitting.
- D) This is data augmentation.

**21. If a linear regression model's residuals show a clear curved pattern when plotted against the predicted values, this indicates a violation of the assumption of:**
- [ ] A) Normality of errors.
- [ ] B) Homoscedasticity (constant variance of errors).
- [ ] C) Independence of errors.
- [ ] D) Linearity.

**Correct Answer:** D

**Explanation:**
- D) A pattern in the residuals suggests that the relationship between the predictors and the outcome variable is not linear, and a simple linear model is not capturing the full complexity of the relationship.

**22. The Minkowski distance with p=1 is equivalent to the:**
- [ ] A) Euclidean distance.
- [ ] B) Chebyshev distance.
- [ ] C) Manhattan distance.
- [ ] D) Cosine similarity.

**Correct Answer:** C

**Explanation:**
- C) When p=1, the Minkowski distance formula simplifies to the sum of the absolute differences between coordinates, which is the definition of the Manhattan distance.

**23. In Stacking, the "meta-model" or "blender" is trained on:**
- [ ] A) The original training data.
- [ ] B) The predictions made by the base models on a hold-out set.
- [ ] C) The residual errors of the base models.
- [ ] D) A random subset of the original features.

**Correct Answer:** B

**Explanation:**
- B) Stacking is a multi-level ensemble method. The base models (Level 0) are trained, and their predictions on a validation set become the features for training the meta-model (Level 1).

**24. A key advantage of a Radial Basis Function (RBF) network over a standard Multi-Layer Perceptron (MLP) is that:**
- [ ] A) RBF networks can have multiple hidden layers, while MLPs can only have one.
- [ ] B) The training process for an RBF network is often much faster due to its hybrid unsupervised/supervised learning approach.
- [ ] C) RBF networks are guaranteed to find a globally optimal solution, whereas MLPs can get stuck in local minima.
- [ ] D) RBF networks do not require an activation function in their hidden layer.

**Correct Answer:** B

**Explanation:**
- B) In a typical RBF network, the hidden layer parameters (centers and spreads) are determined using a fast, unsupervised method like K-Means. Only the final linear output layer requires supervised training, which is generally much faster than the end-to-end backpropagation required for an MLP.

**25. The "exploding gradient" problem in deep neural networks is often mitigated by using:**
- [ ] A) A very high learning rate.
- [ ] B) Dropout.
- [ ] C) Gradient clipping or weight regularization (like L2).
- [ ] D) A linear activation function in all layers.

**Correct Answer:** C

**Explanation:**
- C) Gradient clipping involves capping the magnitude of the gradients during backpropagation to prevent them from becoming too large. Weight regularization adds a penalty to the loss function for large weights, which also helps to keep the gradients in check.

**26. What is the Gini Impurity of a node that contains 10 samples from Class A and 10 samples from Class B?**
- [ ] A) 0
- [ ] B) 0.25
- [ ] C) 0.5
- [ ] D) 1.0

**Correct Answer:** C

**Explanation:**
- The proportions are p(A) = 10/20 = 0.5 and p(B) = 10/20 = 0.5.
- Gini = 1 - (p(A)^2 + p(B)^2) = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 0.5. This is the maximum possible Gini Impurity for a two-class problem.

**27. The Softmax activation function is typically used in the output layer of a neural network for which type of task?**
- [ ] A) Binary classification.
- [ ] B) Multi-class classification.
- [ ] C) Regression.
- [ ] D) Dimensionality reduction.

**Correct Answer:** B

**Explanation:**
- B) The Softmax function takes a vector of raw scores (logits) and converts them into a probability distribution, where each element is between 0 and 1 and the sum of all elements is 1. This is ideal for representing the probabilities of an input belonging to one of several mutually exclusive classes.

**28. In Bayesian Decision Theory, the likelihood ratio compares:**
- [ ] A) The posterior probability of two different classes.
- [ ] B) The prior probability of two different classes.
- [ ] C) The probability of observing the evidence given two different classes, p(x|ω1) / p(x|ω2).
- [ ] D) The conditional risk of two different actions.

**Correct Answer:** C

**Explanation:**
- C) The likelihood ratio is a measure of the strength of the evidence. It quantifies how much more likely the observed feature x is under one class hypothesis (ω1) compared to another (ω2).

**29. Which of the following is a primary advantage of using one-hot encoding over label encoding for categorical features in a linear model?**
- [ ] A) It reduces the number of features in the dataset.
- [ ] B) It is computationally less expensive.
- [ ] C) It avoids introducing an unintended and artificial ordinal relationship between categories.
- [ ] D) It can only be used for binary categorical variables.

**Correct Answer:** C

**Explanation:**
- C) Label encoding assigns arbitrary integers (e.g., 0, 1, 2) to categories. A linear model might interpret these integers as having an order, which is often not true (e.g., is "Paris" > "London"?). One-hot encoding creates separate binary features, removing this artificial ordering.

**30. The "curse of dimensionality" implies that as the number of features increases, the distance between a data point and its true nearest neighbor:**
- [ ] A) Approaches zero.
- [ ] B) Becomes more stable and reliable.
- [ ] C) Approaches the distance to its farthest neighbor, making the concept of "nearest" less meaningful.
- [ ] D) Decreases exponentially.

**Correct Answer:** C

**Explanation:**
- C) In high-dimensional spaces, the volume increases so rapidly that data points become very sparse. As a result, the distances between points become more uniform, and the distinction between the nearest and farthest neighbor diminishes, which poses a problem for distance-based algorithms like k-NN.

**31. The learning rate (η) in the Least-Mean-Square (LMS) algorithm controls:**
- [ ] A) The final error rate of the model.
- [ ] B) The number of epochs required for training.
- [ ] C) The step size of the weight update at each iteration.
- [ ] D) The initial values of the synaptic weights.

**Correct Answer:** C

**Explanation:**
- C) The learning rate is a hyperparameter that scales the magnitude of the weight update. A small learning rate leads to slow but stable convergence, while a large learning rate can lead to faster but potentially unstable learning.

**32. A key property of the eigenvectors of a symmetric matrix (like a covariance matrix) is that they are:**
- [ ] A) Always positive.
- [ ] B) Always parallel to the original axes.
- [ ] C) Always orthogonal to each other.
- [ ] D) Always equal in length.

**Correct Answer:** C

**Explanation:**
- C) A fundamental theorem of linear algebra states that the eigenvectors of a symmetric matrix corresponding to distinct eigenvalues are orthogonal. This property is crucial for PCA, as it ensures the principal components are uncorrelated.

**33. In a Bagging ensemble, the base models are trained on datasets created through:**
- [ ] A) Splitting the original dataset into disjoint subsets.
- [ ] B) Sampling with replacement from the original dataset (bootstrapping).
- [ ] C) Sampling without replacement from the original dataset.
- [ ] D) Augmenting the original dataset with synthetic data.

**Correct Answer:** B

**Explanation:**
- B) Bagging stands for Bootstrap Aggregating. It creates diversity among the base models by training each one on a bootstrap sample, which is a random sample of the original data drawn with replacement.

**34. The Universal Approximation Theorem states that a feedforward network with a single hidden layer can:**
- [ ] A) Solve any computational problem.
- [ ] B) Approximate any continuous function to an arbitrary degree of accuracy.
- [ ] C) Outperform any other machine learning model.
- [ ] D) Learn from unlabeled data.

**Correct Answer:** B

**Explanation:**
- B) This theorem provides the theoretical justification for the power of neural networks. It guarantees that even a relatively simple network architecture is capable of representing a wide variety of complex functions, given enough neurons in the hidden layer.

**35. The main purpose of a confusion matrix is to:**
- [ ] A) Visualize the correlation between different features.
- [ ] B) Summarize the performance of a classification model by showing the counts of correct and incorrect predictions for each class.
- [ ] C) Determine the optimal number of clusters in an unsupervised learning problem.
- [ ] D) Plot the trade-off between a model's bias and variance.

**Correct Answer:** B

**Explanation:**
- B) A confusion matrix provides a detailed breakdown of a classifier's performance, showing the number of true positives, true negatives, false positives, and false negatives, which allows for the calculation of metrics like precision, recall, and accuracy.

**36. A decision tree is considered a "white box" model because:**
- [ ] A) It always produces the most accurate predictions.
- [ ] B) Its internal decision-making logic is transparent and easily interpretable.
- [ ] C) It can only be trained on datasets with a white background.
- [ ] D) It requires no hyperparameters to be tuned.

**Correct Answer:** B

**Explanation:**
- B) The flowchart-like structure of a decision tree allows one to easily follow the path of if-then-else rules that lead to a specific prediction, making it highly interpretable, unlike "black box" models like complex neural networks.

**37. In a regression tree, the prediction for a new data point that falls into a specific leaf node is typically:**
- [ ] A) The majority class of the training samples in that leaf.
- [ ] B) The median of the target values of the training samples in that leaf.
- [ ] C) The average of the target values of the training samples in that leaf.
- [ ] D) The value of the single closest training sample in that leaf.

**Correct Answer:** C

**Explanation:**
- C) A regression tree partitions the feature space into regions, and the prediction for any point within a region (a leaf) is the mean of the target values of the training data that fell into that same region.

**38. The primary goal of the AdaBoost algorithm is to:**
- [ ] A) Reduce the variance of a model by averaging many decorrelated weak learners.
- [ ] B) Combine multiple strong learners into a single, more robust model.
- [ ] C) Sequentially train weak learners, with each new learner focusing on the instances misclassified by the previous ones.
- [ ] D) Use a meta-model to learn the optimal way to combine predictions from a set of diverse base models.

**Correct Answer:** C

**Explanation:**
- C) AdaBoost is a boosting algorithm that works by adaptively re-weighting the training instances. Instances that are misclassified get higher weights, forcing the next weak learner in the sequence to pay more attention to them.

**39. The Chebyshev distance between the points (2, 8) and (7, 3) is:**
- [ ] A) 10
- [ ] B) 7
- [ ] C) 5
- [ ] D) 25

**Correct Answer:** C

**Explanation:**
- The Chebyshev distance is the maximum of the absolute differences between the coordinates.
- Horizontal distance: |7 - 2| = 5
- Vertical distance: |3 - 8| = 5
- The maximum of these two values is 5.

**40. Which activation function is known for potentially causing the "dying ReLU" problem in deep neural networks?**
- [ ] A) Sigmoid
- [ ] B) Tanh
- [ ] C) Leaky ReLU
- [ ] D) ReLU (Rectified Linear Unit)

**Correct Answer:** D

**Explanation:**
- D) The standard ReLU function outputs 0 for any negative input. If a neuron's weights are updated such that its input is always negative, it will always output 0, and the gradient flowing through it will also be 0. This means the neuron effectively "dies" and stops learning.

**41. The Bayes Decision Rule is optimal because it minimizes the:**
- [ ] A) Number of features required.
- [ ] B) Computational complexity.
- [ ] C) Conditional risk for each observation.
- [ ] D) Variance of the model.

**Correct Answer:** C

**Explanation:**
- C) The Bayes Decision Rule states that for any given observation x, one should choose the action (class) that minimizes the conditional risk R(α|x). By making the optimal decision for every single observation, the total average risk (Bayes risk) is minimized.

**42. In PCA, if the first two principal components capture 95% of the total variance, this implies that:**
- [ ] A) The original dataset had only two features.
- [ ] B) The remaining principal components capture the most important information.
- [ ] C) The data can be effectively represented in a 2D space with only a 5% loss of information.
- [ ] D) The original features were perfectly uncorrelated.

**Correct Answer:** C

**Explanation:**
- C) The percentage of variance captured by the principal components is a measure of how much information is retained. Capturing 95% of the variance means that a 2D projection of the data preserves most of its original structure.

**43. A key limitation of a single-layer perceptron is its inability to solve problems that are not:**
- [ ] A) Regression problems.
- [ ] B) Linearly separable.
- [ ] C) Computationally expensive.
- [ ] D) High-dimensional.

**Correct Answer:** B

**Explanation:**
- B) A single-layer perceptron can only learn a linear decision boundary. Therefore, it cannot solve problems like XOR where the classes are not linearly separable.

**44. The process of standardizing data before applying PCA is crucial because:**
- [ ] A) PCA cannot handle negative values.
- [ ] B) It ensures that all principal components will have positive eigenvalues.
- [ ] C) It prevents variables with larger scales from dominating the variance calculation.
- [ ] D) It converts all categorical variables into a numerical format.

**Correct Answer:** C

**Explanation:**
- C) PCA is a variance-based technique. If one feature has a much larger scale (and thus variance) than others, it will disproportionately influence the principal components. Standardization gives all features equal importance by scaling them to have a mean of 0 and a standard deviation of 1.

**45. In a Random Forest, the diversity among the individual trees is primarily achieved by:**
- [ ] A) Using different types of weak learners for each tree.
- [ ] B) Training each tree on the residual errors of the previous one.
- [ ] C) Using both bootstrapping of samples and random selection of features at each split.
- [ ] D) Applying different pruning strategies to each tree.

**Correct Answer:** C

**Explanation:**
- C) Random Forest uses two sources of randomness to decorrelate the trees: 1) Bagging (bootstrapping) ensures each tree is trained on a slightly different dataset, and 2) Feature randomness at each split prevents all trees from relying on the same strong predictors.

**46. The F1-score is the harmonic mean of which two metrics?**
- [ ] A) Accuracy and Recall
- [ ] B) Precision and Recall
- [ ] C) True Positive Rate and False Positive Rate
- [ ] D) Bias and Variance

**Correct Answer:** B

**Explanation:**
- B) The F1-score is calculated as 2 * (Precision * Recall) / (Precision + Recall). It is a useful metric for imbalanced datasets as it provides a balance between precision and recall.

**47. The "bias" term in a neural network neuron allows the activation function to:**
- [ ] A) Be shifted horizontally, increasing the model's flexibility.
- [ ] B) Remain centered at the origin.
- [ ] C) Change its shape from linear to non-linear.
- [ ] D) Control the learning rate of the neuron.

**Correct Answer:** A

**Explanation:**
- A) The bias term adds a constant value to the weighted sum of inputs before it is passed to the activation function. This has the effect of shifting the activation function to the left or right, allowing the neuron to learn a wider range of patterns.

**48. Which of the following is a key assumption of linear regression regarding the residuals (errors)?**
- [ ] A) The residuals should be highly correlated with the predictor variables.
- [ ] B) The residuals should have a mean of zero and a constant variance (homoscedasticity).
- [ ] C) The residuals should follow a U-shaped distribution.
- [ ] D) The variance of the residuals should increase as the predicted value increases.

**Correct Answer:** B

**Explanation:**
- B) Key assumptions for linear regression include that the errors are independent, normally distributed with a mean of 0, and have a constant variance across all levels of the predictor variables.

**49. The primary goal of the backpropagation algorithm is to:**
- [ ] A) Efficiently compute the gradient of the loss function with respect to the network's weights.
- [ ] B) Select the optimal number of hidden layers for a neural network.
- [ ] C) Normalize the input data before it is fed into the network.
- [ ] D) Propagate the input signal forward through the network to make a prediction.

**Correct Answer:** A

**Explanation:**
- A) Backpropagation is the core algorithm used to train feedforward neural networks. It uses the chain rule of calculus to efficiently compute the gradients, which are then used by an optimization algorithm (like SGD) to update the weights and minimize the loss.

**50. A model that performs very well on the training data but poorly on the test data is said to be:**
- [ ] A) Underfitting
- [ ] B) Overfitting
- [ ] C) A good generalizer
- [ ] D) High in bias

**Correct Answer:** B

**Explanation:**
- B) This is the classic definition of overfitting. The model has learned the training data too well, including its noise and specific quirks, and as a result, it fails to generalize to new, unseen data. It has low bias but high variance.
