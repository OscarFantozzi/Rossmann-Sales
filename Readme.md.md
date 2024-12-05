
## Business Problem

Rossmann is a large pharmacy chain operating in over 7 European countries with more than 3,000 stores. 
Sales are influenced by various factors such as competition distance, school and state holidays, location, store type, etc. 
Based on the historical sales data of the stores, the CFO intends to forecast sales for the next six weeks. 
To address this problem, I will process the data and train a Machine Learning (ML) model capable of predicting sales for the next six weeks.

## Data Source:

For this case study, the data was sourced from Kaggle. Kaggle is a renowned platform where various companies 
host competitions by making their data publicly available and reward teams with the best solutions. 
Below is the link to the data source, which can also be found in the "data" folder on GitHub.

https://www.kaggle.com/competitions/rossmann-store-sales/data

## Tools Used:

- Programming Language: Python

## Methodology and Problem-Solving Approach

To solve this problem, I used the cyclic method. The cyclic method is an iterative approach composed of 
several steps that are repeated and improved upon with each iteration to gradually refine the problem's solution. 
The cyclic method prioritizes rapid delivery, allowing for improvements in results and metrics of the ML model after each cycle.

The steps to solve the problem are as follows:

1.0 - Data Description  
2.0 - Feature Engineering  
3.0 - Variable Filtering  
4.0 - Exploratory Data Analysis  
5.0 - Data Preparation  
6.0 - Feature Selection  
7.0 - Machine Learning Modeling  
8.0 - Hyperparameter Fine Tuning  
9.0 - Error Translation and Interpretation  
10.0 - Deployment  

Detailed explanations for each step are provided below.

## STEP 1: Data Description

In a tabular dataset (data in table format), each column represents a phenomenon and each row contains information about that phenomenon. 
Therefore, it is crucial for column names to be intuitive to facilitate understanding of what each observation indicates. 
The Rossmann dataset includes the following columns, which will henceforth be referred to as features:

- **store**: Unique store identifier  
- **day_of_week**: Day of the week in integer format  
- **date**: Date of the sale  
- **sales**: Sales value on a given day  
- **customers**: Number of customers on a given day  
- **open**: Indicator of whether the store was open or closed (0 = closed, 1 = open)  
- **promo**: Indicates whether the store had a promotion on that day  
- **state_holiday**: Indicates if it was a state holiday. Stores typically close on state holidays. 
  Contains: a = public holiday, b = Easter holiday, c = Christmas, 0 = None  
- **school_holiday**: Indicates if the store was affected by school holiday closures on that day  
- **store_type**: Store types, categorized as: a, b, c, d  
- **assortment**: Describes the type of assortment in the store: a = basic, b = extra, c = extended  
- **competition_distance**: Distance in meters to the nearest competitor  
- **competition_open_since_month**: Month when the nearest competitor opened  
- **competition_open_since_year**: Year when the nearest competitor opened  
- **promo2**: Indicates if the store had an extended promotion on the day of the sale. 
  Contains: 0 = store is not participating, 1 = store is participating  
- **promo2_since_week**: Week the store participated in promo2  
- **promo2_since_year**: Year the store participated in promo2  
- **promo_interval**: Consecutive months in which the store had promo2. 
  For example: "Feb,May,Aug,Nov" means the store participated in promo2 during February, May, August, and November of that year.

## STEP 2: Feature Engineering

Feature engineering involves creating new features (columns) from the original dataset or by combining it with another dataset. 
It is common, for instance, to extract features like "Year," "Month," and "Day" from a date in the format 'yyyy-mm-dd' for analyses at finer granularities or to create aggregated tables by Year/Month/Day.

Below are some features that can be extracted from the original dataset or by combining it with other data sources.

![alt text](https://github.com/OscarFantozzi/rossmann_sales/blob/main/img/img_hypothesis.jpg)

In this project, temporal features were extracted, and some categorical features (columns containing textual data) were adjusted. 
During this step, hypotheses were also formulated and will be validated through data and analyses (more details in the notebook rossmann_sales.ipynb).

## STEP 3: Variable Filtering

At this stage, the dataset is formatted with the columns and rows that will be used to train the ML model. 
Essentially, it involves filtering out irrelevant rows and columns at the time of prediction. 

For the rows, only open stores were kept, and the 'customers' feature and other derived features were removed. 
The reason for removing the 'customers' feature (indicating the number of customers in the store on a given day) 
is that during prediction, it is not possible to know how many customers will be in the store.

## STEP 4: Exploratory Data Analysis

In this exploratory data analysis step, the data was analyzed through univariate, bivariate, and multivariate analysis.

### Univariate Analysis

In univariate analysis, the goal is to examine the distribution of dataset features and their characteristics 
without considering other correlated variables. This is important as it helps answer questions like: 
What is the mean value of the feature? What are the minimum and maximum values? 
What type of distribution is observed in the feature? Are there outliers?

In this case, the distribution of the target variable is normal.

### Bivariate Analysis

In bivariate analysis, as the name suggests, the interest lies in observing whether any relationships or patterns exist 
between the features and the target variable. Initially, some hypotheses were formulated and answered using feature relationships. 

For example, to answer the hypothesis: "Do stores with larger assortments sell more?" 
the relationship between the "assortment" and "sales" features was analyzed to observe patterns.

As analyzed, the hypothesis is false, meaning stores with "basic" assortments sell more.  
![alt text](https://github.com/OscarFantozzi/rossmann_sales/blob/main/img/analise_bivariada.jpg)

## Multivariate Analysis

In multivariate analysis, Pearson's correlation method was applied to identify correlations between numerical variables. 
For categorical variables (features containing text), Cramér's V was used (more details in the Jupyter notebook).

## STEP 5: Data Preparation

The learning process for most Machine Learning algorithms is facilitated by **numerical data** and data on the same **scale**. 
It is crucial for data to have the same scale because many algorithms tend to prioritize features with larger ranges, 
thus biasing the model. Keeping this in mind, the following steps were applied:

### Normalization

Normalization is used to bring data into the same range. It rescales the data with a center of 0 and a standard deviation of 1. 
The standard deviation, in simple terms, is the measure of how far a given value is from the mean. 

For instance, if the "competition_distance" feature has a mean of 100 meters and a standard deviation of 20, 
the range is 100 ± 20 (80 to 120). After normalization, the scale becomes centered around 0 ± 1.

Normalization works well for normal distributions. However, since the data does not have normal distributions in this case, normalization was not applied.

Normalization follows the formula below:

![alt text](https://github.com/OscarFantozzi/rossmann_sales/blob/main/img/normalizacao.jpg)

Essentially, the mean of a feature (or dataset column) is calculated, and for each value, the mean is subtracted and divided by the standard deviation. This changes the scale of the feature.

### Rescaling

Rescaling adjusts the scale for features that do not have normal distributions. It rescales the values to the interval between 0 and 1.

### Transformation

- **Encoding**: Conversion of categorical features to numerical features.  
- **Nature Transformation**: Applied to cyclic variables that repeat themselves. 
  For example, a year exhibits cyclic behavior, as every 12 months, another year begins.

## STEP 6: Feature Selection

Feature selection involves identifying the most relevant variables that influence the target variable (sales). This helps to reduce noise and improve the model's performance. In this project:
- Features with low correlation to the target variable or high multicollinearity were removed.
- Principal Component Analysis (PCA) was considered to reduce dimensionality, though categorical features required separate handling.

## STEP 7: Machine Learning Modeling

The modeling step involved training different machine learning algorithms to forecast sales for the next six weeks. The steps included:
- Splitting the dataset into training and testing sets.
- Evaluating models like Random Forest, Gradient Boosting, and XGBoost.
- XGBoost showed the best performance with the default parameters, capturing the relationships between features effectively.

## STEP 8: Hyperparameter Fine Tuning

Hyperparameter tuning was conducted to optimize the XGBoost model further. This was done using grid search and random search, focusing on parameters such as:
- `learning_rate`: Controls how much the model adjusts weights with each training step.
- `n_estimators`: Number of trees to grow.
- `max_depth`: Maximum depth of a tree to prevent overfitting.
- `min_child_weight`: Minimum sum of weights required for a node split.

The best parameters improved the model's predictive accuracy and reduced errors.

## STEP 9: Error Translation and Interpretation

After training, the model's errors were analyzed to ensure predictions made sense. Metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) were used. Additionally:
- Residuals were plotted to check for patterns that the model might have missed.
- Predicted vs. actual sales graphs helped in interpreting the overall model behavior.

## STEP 10: Deployment

The final step was deploying the model so that predictions could be made dynamically. Key steps included:
- Creating a Flask application to serve predictions via an API.
- The `rossmann_bot.py` script connects the Telegram bot to the deployed model.
- Users input store IDs, and the bot returns the predicted sales for the next six weeks.

Deployment allowed for real-time interaction, making the solution accessible and efficient for stakeholders.
