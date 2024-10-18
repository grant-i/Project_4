# Nutritional Price Prediction Project

This project aims to create a predictive model that estimates the pricing of food products based on their nutritional content. The main goal is to understand the relationship between nutrient values and price, and evaluate whether certain foods are overpriced or underpriced.

## Project Structure

- **Data Preprocessing**: 
  - The dataset used in this project was loaded from a `.tsv` file.
  - Various cleaning operations were applied, such as standardizing nutritional information and handling missing values.

- **Modeling**: 
  - Multiple machine learning models were developed and tested to predict prices, including:
    - **Linear Regression**
    - **Random Forest**
    - **Principal Component Analysis (PCA)** was used to reduce dimensionality and highlight the most important features.

## Key Features

The dataset contains several engineered features that were helpful in normalizing the dataset from [Kaggle](https://www.kaggle.com/code/allunia/hidden-treasures-in-our-groceries):

- `transformed_carbohydrates_100g`
- `transformed_fat_100g`
- `transformed_proteins_100g`
- `transformed_sugars_100g`
- `transformed_salt_100g`
- `transformed_other_carbs`
- `transformed_energy_100g`
- `transformed_g_sum`
- `reconstructed_energy`


Second and Third Datasets [USDA fruit and vegetables](https://www.ers.usda.gov/data-products/fruit-and-vegetable-prices/) // [USDA other](https://www.ers.usda.gov/data-products/purchase-to-plate/):
-  `product prices`

### Feature Details:
- **g_sum**: Represents the rounded sum of the fat, carbohydrates, proteins, and salt values in the data. This helps identify products with potential false entries.
- **reconstructed_energy**: Calculates the energy value of a product based on fat, carbohydrates, and proteins. It is compared with the dataset's given energy values to check for discrepancies.

These features contribute to the understanding of a food product's nutritional profile and play a crucial role in predictive modeling.

## Performance Metrics

The performance of the models was evaluated using metrics such as R² and Mean Squared Error (MSE):

**Linear Regression**
Mean Squared Error: 12.495409510694495
*R-squared: 0.43026604968281645*

**Random Foreset Regression**
Mean Squared Error: 13.950298154366925
*R-squared: 0.36392973205179613*

**Linear Regression w/ Reduced Features**
Mean Squared Error (Reduced Features): 11.850553551899461
*R-squared (Reduced Features): 0.4596685540565566*

**PCA**
Mean Squared Error: 10.800929008597786
*R-squared: 0.5075266684219414*


## Installation & Dependencies

To run the project locally, follow these steps:

1. Clone the repository:
```
   git clone https://github.com/grant-i/Project_4
```

2. Install required dependencies:
   
```
   https://github.com/grant-i/Project_4/blob/main/requirements.txt
```

## Data Loading, Standardization, and Cleaning

### Data Format 

1. Loading the Data

Here's an example of loading the datasets used in the project:

```
def load_data(file_path):
    path = Path(file_path)
    if path.is_file():
        return pd.read_csv(path, encoding='ISO-8859-1')
    else:
        print(f"Error: {file_path} not found.")
        return pd.DataFrame()
```
Load datasets
```
    veg_df = load_data('part 1 ETL Workflow/raw_files/Vegetable-Prices-2022.csv')
    fru_df = load_data('part 1 ETL Workflow/raw_files/Fruit-Prices-2022.csv')
    more_df = load_data('part 1 ETL Workflow/raw_files/pp_national_average_prices_csv.csv')
    off_df = load_data('part 1 ETL Workflow/raw_files/hidden_treasures_groceries_gmm.csv')
```

2. Standardizing the Data

Standardization ensures consistency across units:

```
inflation_factor = 2
grams_per_pound = 453.592
price_per_100g_to_lb = grams_per_pound / 100

more_df = more_df.assign(
    RetailPrice=more_df['price_100gm'] * inflation_factor * price_per_100g_to_lb
).drop(columns=['price_100gm'])
```

3. Merge, Clean and Save

```
working_df = pd.concat([vegnut_df, fruitnut_df, morenut_df], ignore_index=True)

# Step 1: Dropping unnecessary columns such as 'Unnamed: 0'
df_cleaned = working_df.drop(columns=['Unnamed: 0'])

# Step 2: Check for missing values
missing_values = df_cleaned.isnull().sum()

# Step 3: Drop rows with missing target values (if any)
df_cleaned = df_cleaned.dropna(subset=['RetailPrice'])
```
# Save the dataset to CSV

```
df = pd.read_csv('final_working.csv')
df.head()
```

4. Visually Inspect Data


** mean for transformed columns **
These are normalized values from the kaggle data set
```
Averages of features in df:
transformed_carbohydrates_100g      0.041203
transformed_fat_100g               -0.184333
transformed_proteins_100g          -0.346366
transformed_sugars_100g             0.132937
transformed_salt_100g              -0.207586
transformed_other_carbs            -0.070118
transformed_energy_100g            -0.039228
transformed_reconstructed_energy   -0.032001
transformed_g_sum                  -0.024871
```



# Graphs

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/histogram_1.png) ![alt text](https://github.com/grant-i/Project_4/blob/main/figures/scatter_1.png)

# Shape of Data

```# Remove rows where RetailPrice is greater than 30
df_no_outliers = df[df['RetailPrice'] <= 30]
```

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/hist_2.png) ![alt text](https://github.com/grant-i/Project_4/blob/main/figures/scat_2.png)


** Comparison of removal of outliers **
There is a lot of change after removing the outliers.

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/super_heat.png)


# Inspect Features
```
# Correlation of Data
sns.heatmap(df_essentials.corr(), annot=True, cmap='coolwarm')
plt.show()
```

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/heat_map.png)




5. Model Data

```
# Prepare the data for training
line_X = df_essentials[features]
line_y = df_essentials[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(line_X, line_y, test_size=0.2, random_state=42)
```
NOTE: After feature analysis a new variable y was declared instead of line_y for reduced data frame
```X_reduced = df_no_outliers[reduced_features]
y = df_no_outliers[target]
```

# Linear Regression
```# Create and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the test set
line_y_pred = linear_model.predict(X_test)

# Calculate performance metrics
line_mse = mean_squared_error(y_test, line_y_pred)
line_r2 = r2_score(y_test, line_y_pred)

# Display the coefficients, MSE, and R-squared value
line_coefficients = linear_model.coef_
line_intercept = linear_model.intercept_

print("Mean Squared Error:", line_mse)
print("R-squared:", line_r2)

line_coefficients, line_intercept
```

Mean Squared Error: 12.495409510694495
R-squared: 0.43026604968281645


![alt text](https://github.com/grant-i/Project_4/blob/main/figures/linear_reg_!.png)

Maybe a poor fit because of clusters


![alt text](https://github.com/grant-i/Project_4/blob/main/figures/scatter_cluster.png)



Random Forest is supposed to be better at seeing clusters especially since linear regression is made for lines.
# Random Forest Regression

```
# Prepare the data for training
forest_X = df_essentials[features]
forest_y = df_essentials[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(forest_X, forest_y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

Random Forest R-squared: 0.36392973205179613 compared to Linear Regression R-squared: 0.43026604968281645

Random Forest Mean Squared Error: 13.950298154366925 compared to Linear Regression Mean Squared Error: 12.495409510694495

**No Improvement**

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/forest_FI.png)

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/cor_RP.png)

```
                             Feature   VIF

0     transformed_carbohydrates_100g   39.786989
1               transformed_fat_100g   10.062862
2          transformed_proteins_100g    2.560955
3            transformed_sugars_100g    4.677926
4              transformed_salt_100g    1.858409
5            transformed_other_carbs    5.595568
6            transformed_energy_100g   95.999928
7   transformed_reconstructed_energy  230.609919
8                  transformed_g_sum  136.725746
9                        RetailPrice    1.939766
10                          constant    6.251324
```


# Feature Selection and Model Validation
```# Create a copy of the features list to avoid modifying the original
reduced_features = features.copy()

# Remove the unwanted features
reduced_features.remove('transformed_sugars_100g')
reduced_features.remove('transformed_reconstructed_energy')

print(reduced_features)
```

# Linear Regression 

```
X_reduced = df_no_outliers[reduced_features]
y = df_no_outliers[target]
```

Linear Regression Mean Squared Error: 12.495409510694495
Linear Regression R-squared: 0.43026604968281645

Mean Squared Error (**Reduced Features**): 11.850553551899461
R-squared (**Reduced Features**): 0.4596685540565566

*Improvement*

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/line_compare.png)


# PCA

```
# Standardizing the data
scaler = StandardScaler()
line_X_scaled = scaler.fit_transform(line_X)

# Apply PCA
pca = PCA(n_components=0.95)  # Retaining components explaining 95% of the variance
line_X_pca = pca.fit_transform(line_X_scaled)

# Check how many components were retained
pca.n_components_

# Standardizing the data
scaler = StandardScaler()
line_X_scaled = scaler.fit_transform(line_X)

# Apply PCA to reduce to 5 components
pca = PCA(n_components=5)
line_X_pca = pca.fit_transform(line_X_scaled)

# Check the shape of the transformed data
print("Shape of the data after PCA:", line_X_pca.shape)
```
Shape of the data after PCA: (118, 5)

```
# Split the data into training and testing sets again
X_train_pca, X_test_pca, y_train, y_test = train_test_split(line_X_pca, line_y, test_size=0.2, random_state=42)

# Create and train the linear regression model with the reduced feature set
model_reduced = LinearRegression()
model_reduced.fit(X_train_pca, y_train)

# Predicting the test set results
y_pred_pca = model_reduced.predict(X_test_pca)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_pca)
r2 = r2_score(y_test, y_pred_pca)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

**Mean Squared Error: 10.800929008597786** 
**R-squared: 0.5075266684219414**

*Improvement*

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/pca_viz_1.png)

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/pca_viz_2.png)



## Conclusion



## Afterwords 
Additional Feature Engineering 

**Feature Engineering of Protein to Carbohydrate Ratio**

```# Create protein-to-carb ratio and drop the original 'proteins_100g' and 'carbohydrates_100g' columns if needed
df_with_eng_feature = df_no_outliers.assign(
    protein_to_carb_ratio=df_no_outliers['proteins_100g'] / df_no_outliers['carbohydrates_100g']
).replace([np.inf, -np.inf], np.nan).fillna(0).drop(columns=['proteins_100g', 'carbohydrates_100g'])

eng_f = ['transformed_carbohydrates_100g']
```
** Linear Regression for Engineered Features **
```
Mean Squared Error: 11.850553551899438
R-squared: 0.4596685540565577
```
**Overall Poor Performance of Linear Regression Model**

Test Set Performance:
Mean Squared Error (Test): 11.850553551899438
R-squared (Test): 0.4596685540565577

Training Set Performance:
Mean Squared Error (Train): 10.369893741854172
R-squared (Train): 0.4725794249144787

**Additional PCA of Engineered Feature**
Mean Squared Error: 11.077177568030576
R-squared: 0.49493098815228953

**Redraft**
Begin with IQR and Z-score filters on a larger data set and then reduce
