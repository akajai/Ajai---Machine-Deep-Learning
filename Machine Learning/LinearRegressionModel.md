<div style="text-align: justify;">

## Linear Model For Regression

Linear regression is a way to predict an unknown value by finding the simplest possible relationship between things you already know. 

- Goal: Is to use the inputs to predict the values of the outputs.
- Input: Independent Variables.
- Output: Response/Dependent Variable.
- Qualitative: Information that is descriptive and conceptual, dealing with qualities
- Quantitative: Information that can be counted or measured, dealing with numbers.
- Prediction Tasks: Regression (quantitative outputs) vs Classification (qualitative outputs).

#### Different Encoding Qualitative Outputs: Label encoding vs. One-hot encoding.

Encoded qualitative outputs are categorical labels (like "cat," "dog," or "bird") that have been converted into a numerical format so that machine learning algorithms can process them.

Why do we need to encode them?

Machine learning models are mathematical, which means they work with numbers, not text. An algorithm can't directly understand the word "Delhi" or "spam." Before we can train a model to predict a qualitative output, we must first translate those word-based categories into numbers.

**Common Encoding Methods**

1. **Label encoding** assigns a unique integer to each category in a feature. For example, if you have a "Colour" feature with categories 'Red', 'Green', and 'Blue', label encoding would convert them like this.

    | Original | Encoded |
    |---|---|
    | Red | 0 |
    | Green | 1 |
    | Blue | 2 |

2. **One-hot encoding** takes a categorical feature and creates a new binary (0 or 1) column for each unique category. For each row, a '1' is placed in the column corresponding to its category, and '0's are placed everywhere else

    | Original | Red | Green | Blue |
    |---------|---|-----|----|
    | Red      |  1  |   0   |  0   |
    | Green    |  0  |   1   |  0   |
    | Blue     |  0  |   0   |  1   |



### Previous Topic --> [Pattern Recognition](./PatternReocognition.md) 
### Next Topic --> [Linear Regression](./LinearRegressionModel.md)

</div>