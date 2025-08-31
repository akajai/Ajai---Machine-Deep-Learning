# Logistic Regression Quiz

Here are 20 multiple-choice questions based on the Logistic Regression notes.

### Questions

1.  **What is the primary purpose of logistic regression?**
    *   [ ] A) To reduce the number of features in a dataset.
    *   [ ] B) To classify data into two or more categories.
    *   [ ] C) To group similar data points together.
    *   [ ] D) To predict a continuous value.

    **Answer: B) To classify data into two or more categories.**

    **Explanation:**
    *   A is incorrect. Reducing features is the purpose of dimensionality reduction techniques.
    *   B is correct. Logistic regression is a classification algorithm used to predict a binary (or multiclass) outcome.
    *   C is incorrect. Grouping similar data points is the goal of clustering algorithms.
    *   D is incorrect. Predicting a continuous value is the purpose of linear regression.

2.  **What type of function does logistic regression use to model the probability of an outcome?**
    *   [ ] A) A step function.
    *   [ ] B) A quadratic function.
    *   [ ] C) A sigmoid function.
    *   [ ] D) A linear function.

    **Answer: C) A sigmoid function.**

    **Explanation:**
    *   A is incorrect. A step function would create a hard classification boundary, not a probability.
    *   B is incorrect. A quadratic function is not suitable for modeling probabilities.
    *   C is correct. The sigmoid (or logistic) function is an "S"-shaped curve that maps any real-valued number into a value between 0 and 1, which is ideal for representing probability.
    *   D is incorrect. A linear function would produce values outside the 0-1 probability range.

3.  **If the output of the sigmoid function is 0.8, what does this signify?**
    *   [ ] A) The input value is 0.8.
    *   [ ] B) There is an 80% chance that the input belongs to the positive class (class 1).
    *   [ ] C) The model is 80% accurate.
    *   [ ] D) There is an 80% chance that the input belongs to the negative class (class 0).

    **Answer: B) There is an 80% chance that the input belongs to the positive class (class 1).**

    **Explanation:**
    *   A is incorrect. The output is the probability, not the input value itself.
    *   B is correct. The output of the sigmoid function in logistic regression is the estimated probability of the positive class (usually denoted as 1).
    *   C is incorrect. The output is a prediction for a single data point, not the overall model accuracy.
    *   D is incorrect. The output represents the probability of the positive class.

4.  **In the context of a spam filter, what would be considered a "feature" for a logistic regression model?**
    *   [ ] A) The logistic regression algorithm itself.
    *   [ ] B) The number of times the word "free" appears in the email.
    *   [ ] C) The entire collection of emails used for training.
    *   [ ] D) The final classification of an email as "spam" or "not spam".

    **Answer: B) The number of times the word "free" appears in the email.**

    **Explanation:**
    *   A is incorrect. The algorithm is the model being used.
    *   B is correct. Features (or independent variables) are the input characteristics of the data used to make a prediction, such as the frequency of certain words.
    *   C is incorrect. The collection of emails is the dataset.
    *   D is incorrect. The classification is the output (dependent variable), not a feature.

5.  **If the probability of an event happening is 0.75, what are the odds of the event happening?**
    *   [ ] A) 1 to 4
    *   [ ] B) 3 to 1
    *   [ ] C) 4 to 1
    *   [ ] D) 1 to 3

    **Answer: B) 3 to 1**

    **Explanation:**
    *   The probability of the event happening is P(event) = 0.75.
    *   The probability of the event not happening is P(not event) = 1 - 0.75 = 0.25.
    *   Odds = P(event) / P(not event) = 0.75 / 0.25 = 3.
    *   This is expressed as odds of 3 to 1.

6.  **What does an odds ratio of 5 mean?**
    *   [ ] A) The odds of an event happening in one group are 5 times the odds of it happening in another group.
    *   [ ] B) For every 5 times the event occurs, it fails to occur 1 time.
    *   [ ] C) The event is 5% likely to occur.
    *   [ ] D) The probability of an event is 5.

    **Answer: A) The odds of an event happening in one group are 5 times the odds of it happening in another group.**

    **Explanation:**
    *   A is correct. The odds ratio is a ratio of two odds, used to compare the likelihood of an event in two different groups.
    *   B describes odds of 5 to 1, not an odds ratio.
    *   C is incorrect. An odds ratio is not a direct percentage.
    *   D is incorrect. Probability cannot be greater than 1.

7.  **What does the coefficient vector (β) in a logistic regression model represent?**
    *   [ ] A) The final classification of the data.
    *   [ ] B) The predicted probabilities for each data point.
    *   [ ] C) The weights assigned to each feature, indicating their importance and direction of effect.
    *   [ ] D) The input features of the model.

    **Answer: C) The weights assigned to each feature, indicating their importance and direction of effect.**

    **Explanation:**
    *   A is incorrect. The final classification is the outcome based on the probabilities.
    *   B is incorrect. The predicted probabilities are the output of the model.
    *   C is correct. The coefficient vector contains the parameters (weights) that the model learns during training. Each coefficient corresponds to a feature.
    *   D is incorrect. The input features are represented by X.

8.  **What is the primary goal of Maximum Likelihood Estimation (MLE)?**
    *   [ ] A) To find the model parameters that make the observed data most probable.
    *   [ ] B) To calculate the p-value of the model.
    *   [ ] C) To find the straight line that best fits the data.
    *   [ ] D) To minimize the number of features in the model.

    **Answer: A) To find the model parameters that make the observed data most probable.**

    **Explanation:**
    *   A is correct. MLE is a method for estimating the parameters of a statistical model by finding the parameter values that maximize the likelihood function.
    *   B is incorrect. While related to statistical inference, calculating the p-value is not the primary goal of MLE.
    *   C is incorrect. This describes Ordinary Least Squares (OLS) for linear regression.
    *   D is incorrect. This is the goal of feature selection or dimensionality reduction.

9.  **How is Ordinary Least Squares (OLS) different from the method used to fit a logistic regression model?**
    *   [ ] A) OLS maximizes the likelihood, while logistic regression minimizes it.
    *   [ ] B) OLS minimizes the sum of squared residuals to fit a line, which is not suitable for the probabilistic nature of logistic regression.
    *   [ ] C) There is no difference; OLS is used for both.
    *   [ ] D) OLS is used for classification, while logistic regression is for regression.

    **Answer: B) OLS minimizes the sum of squared residuals to fit a line, which is not suitable for the probabilistic nature of logistic regression.**

    **Explanation:**
    *   A is incorrect. OLS minimizes squared error, while logistic regression (via MLE) maximizes likelihood.
    *   B is correct. OLS is used for linear regression. Logistic regression uses Maximum Likelihood Estimation (MLE) because its output is a probability, and the relationship is non-linear (due to the sigmoid function).
    *   C is incorrect. They use different optimization methods.
    *   D is incorrect. The roles are reversed.

10. **In machine learning, what is a cost function?**
    *   [ ] A) The number of features used in the model.
    *   [ ] B) A measure of how wrong a model's predictions are.
    *   [ ] C) The computational resources required to train a model.
    *   [ ] D) The price of the software used to build the model.

    **Answer: B) A measure of how wrong a model's predictions are.**

    **Explanation:**
    *   A is incorrect. This is the dimensionality of the feature space.
    *   B is correct. A cost function (or loss function) quantifies the error between predicted values and actual values. The goal of training is to minimize this function.
    *   C is incorrect. While related to efficiency, this is not the definition of a cost function.
    *   D is incorrect. It's a mathematical concept, not a financial one.

11. **How is the log-likelihood related to the cost function in logistic regression?**
    *   [ ] A) The cost function is the exponential of the log-likelihood.
    *   [ ] B) The cost function is the negative of the log-likelihood.
    *   [ ] C) They are unrelated concepts.
    *   [ ] D) The cost function is the log-likelihood.

    **Answer: B) The cost function is the negative of the log-likelihood.**

    **Explanation:**
    *   A is incorrect. This would be the likelihood itself, not the cost function.
    *   B is correct. To frame the problem of maximizing the log-likelihood as a minimization problem (which is standard for cost functions), we use the negative log-likelihood. Minimizing the negative log-likelihood is equivalent to maximizing the log-likelihood. This is also known as Log Loss or Cross-Entropy Loss.
    *   C is incorrect. They are directly related.
    *   D is incorrect. Likelihood is maximized, while cost is minimized.

12. **What is a decision boundary?**
    *   [ ] A) The line where the probability of belonging to a class is exactly 0.5.
    *   [ ] B) The algorithm's accuracy score.
    *   [ ] C) The dataset used to train the model.
    *   [ ] D) The final output of the logistic regression model.

    **Answer: A) The line where the probability of belonging to a class is exactly 0.5.**

    **Explanation:**
    *   A is correct. The decision boundary is the separating line or surface that the model learns. For logistic regression, this is the threshold (typically p=0.5) where the model is uncertain and would classify a point into either class.
    *   B is incorrect. The accuracy score is a metric to evaluate the model.
    *   C is incorrect. The dataset is the input data.
    *   D is incorrect. The final output is a classification or probability.

13. **Can a decision boundary in logistic regression be non-linear?**
    *   [ ] A) No, non-linear boundaries are only possible in linear regression.
    *   [ ] B) Yes, but only if the model has more than two features.
    *   [ ] C) Yes, if you create polynomial features or use a more complex model.
    *   [ ] D) No, it is always a straight line.

    **Answer: C) Yes, if you create polynomial features or use a more complex model.**

    **Explanation:**
    *   A is incorrect. Linear regression does not have a decision boundary as it predicts continuous values.
    *   B is incorrect. The number of features determines the dimensionality of the boundary, not its shape.
    *   C is correct. By engineering features (e.g., adding x², x³, or interaction terms), you can create a non-linear decision boundary with a logistic regression model.
    *   D is incorrect. While the default decision boundary for logistic regression is linear, it can be made non-linear.

14. **What is the purpose of the One-vs-Rest (OvR) strategy?**
    *   [ ] A) To select the most important features for the model.
    *   [ ] B) To adapt binary logistic regression for multiclass classification problems.
    *   [ ] C) To reduce the cost function of the model.
    *   [ ] D) To improve the accuracy of binary logistic regression.

    **Answer: B) To adapt binary logistic regression for multiclass classification problems.**

    **Explanation:**
    *   A is incorrect. This is the purpose of feature selection methods.
    *   B is correct. OvR is a technique to handle multiclass problems by training N separate binary classifiers, where N is the number of classes. Each classifier distinguishes one class from all the others.
    *   C is incorrect. It's a strategy for problem formulation, not optimization.
    *   D is incorrect. It doesn't inherently improve binary classification.

15. **How does the Softmax function differ from the sigmoid function?**
    *   [ ] A) There is no difference; they are the same function.
    *   [ ] B) Softmax outputs a probability distribution over multiple classes that sums to 1, while sigmoid outputs a single probability for one class.
    *   [ ] C) Softmax is a linear function, while sigmoid is non-linear.
    *   [ ] D) Softmax is used for binary classification, while sigmoid is for multiclass.

    **Answer: B) Softmax outputs a probability distribution over multiple classes that sums to 1, while sigmoid outputs a single probability for one class.**

    **Explanation:**
    *   A is incorrect. They are different functions.
    *   B is correct. The sigmoid function is a special case of the softmax function for K=2 classes. Softmax generalizes this to handle K > 2 classes, providing a vector of probabilities.
    *   C is incorrect. Both are non-linear functions.
    *   D is incorrect. The roles are reversed.

16. **If a logistic regression model has a large positive coefficient for a feature, what does this imply?**
    *   [ ] A) The feature has a negative correlation with the outcome.
    *   [ ] B) An increase in the value of that feature decreases the probability of the outcome being in the positive class.
    *   [ ] C) An increase in the value of that feature increases the probability of the outcome being in the positive class.
    *   [ ] D) The feature is not important for the prediction.

    **Answer: C) An increase in the value of that feature increases the probability of the outcome being in the positive class.**

    **Explanation:**
    *   A is incorrect. It implies a positive correlation.
    *   B is incorrect. A positive coefficient means a positive relationship with the log-odds of the outcome.
    *   C is correct. A positive coefficient means that as the feature value increases, the log-odds of the event occurring increase, which in turn increases the probability.
    *   D is incorrect. A large coefficient (positive or negative) indicates high importance.

17. **What is the typical threshold used to classify the output probability in binary logistic regression?**
    *   [ ] A) It depends on the number of features.
    *   [ ] B) 1.0
    *   [ ] C) 0.5
    *   [ ] D) 0.0

    **Answer: C) 0.5**

    **Explanation:**
    *   A is incorrect. The threshold is independent of the number of features.
    *   B and D are incorrect. These would mean classifying everything as one class.
    *   C is correct. A standard threshold of 0.5 is used. If the predicted probability is > 0.5, the instance is classified as the positive class (1); otherwise, it's classified as the negative class (0). This threshold can be adjusted based on the problem's needs.

18. **Which of the following is a key assumption of logistic regression?**
    *   [ ] A) The variance of the residuals is constant.
    *   [ ] B) The relationship between the features and the log-odds of the outcome is linear.
    *   [ ] C) The features must be perfectly uncorrelated.
    *   [ ] D) The features must be normally distributed.

    **Answer: B) The relationship between the features and the log-odds of the outcome is linear.**

    **Explanation:**
    *   A is incorrect. This is an assumption of linear regression (homoscedasticity), not logistic regression.
    *   B is correct. This is the core assumption. The logistic function transforms a linear combination of inputs into a probability.
    *   C is incorrect. While high multicollinearity can be a problem, it's not a strict assumption that they are perfectly uncorrelated.
    *   D is incorrect. Logistic regression does not assume normality of features.

19. **In the One-vs-Rest (OvR) method for a 4-class problem, how many binary classifiers are trained?**
    *   [ ] A) 6
    *   [ ] B) 2
    *   [ ] C) 4
    *   [ ] D) 1

    **Answer: C) 4**

    **Explanation:**
    *   The One-vs-Rest (OvR) method trains one classifier per class. Since there are 4 classes, it will train 4 separate binary classifiers (Class 1 vs. Rest, Class 2 vs. Rest, Class 3 vs. Rest, Class 4 vs. Rest).

20. **What is another name for the cost function used in logistic regression?**
    *   [ ] A) Hinge Loss
    *   [ ] B) Mean Absolute Error
    *   [ ] C) Log Loss or Cross-Entropy Loss
    *   [ ] D) Sum of Squared Errors

    **Answer: C) Log Loss or Cross-Entropy Loss**

    **Explanation:**
    *   A is incorrect. Hinge Loss is typically used for Support Vector Machines (SVMs).
    *   B is incorrect. This is another regression metric.
    *   C is correct. The cost function derived from the negative log-likelihood is commonly known as Log Loss or Cross-Entropy Loss in the context of classification.
    *   D is incorrect. This is the cost function for linear regression.

    ### Back to Reading Content --> [Logistic Regression Model](../LogisticRegression.md)