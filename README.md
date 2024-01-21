# RiceDatathon2024

## Inspiration
Given the challenge of predicting peak oil production rates, we initially focused on data cleaning, which involved removing rows without a final production rate and columns missing more than 80% of their data. Our first approach was to build a linear regression model to get a sense of the dataset. After achieving RMSEs between 105-120 with various cleaned datasets, we explored different machine learning models.

## What it does
We chose the Random Forest algorithm, an ensemble learning method known for its effectiveness in regression and classification tasks. For regression tasks like predicting peak oil production, it constructs multiple decision trees during training and outputs the average of these trees' predictions. This method helps reduce noise and variance, leading to more accurate results.

**Advantages:**
1) High Accuracy
2) Overfitting Resistance 
3) Handles Various Data Types

## How we built it
We began by preprocessing the dataset, handling both categorical and numerical features. Categorical features were one-hot encoded to ensure proper processing by the model. The next step involved model training and extensive hyperparameter tuning using GridSearchCV, allowing us to systematically explore various parameters and optimize the model for reduced RMSE.

## Challenges we ran into
1. We encountered issues with missing data, eventually choosing to eliminate all rows with missing values after getting rid of less informative columns. Fortunately, we still retained over 13,000 rows.
2. We faced high RMSEs with initial attempts using models like multilinear regression, FNN, CNN, and gradient boosting regressors.


## Accomplishments that we're proud of
1, Our initial data analysis involved using clustering and exploring median versus mean imputation to determine our data cleaning strategy, which was highly educational.
2, We built and evaluated multiple machine learning models, basing our final decision on the calculated RMSE.
3. Our team demonstrated excellent collaboration and teamwork.

## What we learned
We learned how much impact different ways of data cleansing can have on the same model, methods to optimize models by adjusting parameters, and of course how oil production works! 

## What's next for Essential Oil: Predicting Peak Oil Production
Our first attempt in this datathon of finding relationship between different column parameters didn't provide much insight, primarily because we focused on exploring linear relationship. In the future, we will continue to explore other kinds of relationship between different parameters to optimize our data imputing process and optimize our model. 
