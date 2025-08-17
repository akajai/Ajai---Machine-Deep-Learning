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



**Example of Linear Model**

Let's say you own an ice cream shop and want to predict how many ice creams you'll sell based on the day's temperature. You have the following data from the last few days:
        
- At 25°C, you sold 50 ice creams.
- At 30°C, you sold 60 ice creams.
- At 35°C, you sold 70 ice creams.
        
You can plot these points on a graph. You'll notice they don't form a perfect straight line, but there's a clear trend: higher temperature means more sales.
        
**Your Goal**: You want to draw one straight line through these points to predict tomorrow's sales. If the weather forecast says it will be 31°C, how many ice creams should you prepare?
        
**How Least Squares Helps**: Instead of just eyeballing it, the method of least squares will mathematically calculate the perfect line. It finds the line where the sum of the squared distances (the errors) between what you actually sold each day and what the line predicts you would sell is minimized.
        
**Intercept (The Starting Point)**: Think of this as your baseline sales. It's the number of ice creams you'd probably sell even on a very cold day (let's say 0°C), maybe to a few die-hard fans. Let's say your intercept is 10.

**Slope (The "Effect" of Temperature)**: This is the most important part. It's the magic number that tells you how many extra ice creams you sell for every 1-degree increase in temperature. Let's say you find the slope is 2.

With this model, you can now predict sales for any temperature. If the forecast says it will be 32°C tomorrow, you can predict you'll sell (2 * 32) + 10 = 74 ice creams.


### Previous Topic --> [Pattern Recognition](./PatternReocognition.md) 
### Next Topic --> [Linear Regression](./LinearRegressionModel.md)

</div>