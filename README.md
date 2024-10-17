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

The dataset contains several engineered features that were helpful in normalizing the dataset from Kaggle. [Dataset link](https://www.kaggle.com/code/allunia/hidden-treasures-in-our-groceries):

- `transformed_carbohydrates_100g`
- `transformed_fat_100g`
- `transformed_proteins_100g`
- `transformed_sugars_100g`
- `transformed_salt_100g`
- `transformed_other_carbs`
- `transformed_energy_100g`
- `transformed_g_sum`
- `reconstructed_energy`

### Feature Details:
- **g_sum**: Represents the rounded sum of the fat, carbohydrates, proteins, and salt values in the data. This helps identify products with potential false entries.
- **reconstructed_energy**: Calculates the energy value of a product based on fat, carbohydrates, and proteins. It is compared with the dataset's given energy values to check for discrepancies.

These features contribute to the understanding of a food product's nutritional profile and play a crucial role in predictive modeling.

## Performance Metrics

The performance of the models was evaluated using metrics such as R² and Mean Squared Error (MSE):

**Linear Regression**
Mean Squared Error: 12.495409510694495
R-squared: 0.43026604968281645

**Random Foreset Regression**
Mean Squared Error: 13.950298154366925
R-squared: 0.36392973205179613

**Linear Regression w/ Reduced Features**
Mean Squared Error (Reduced Features): 11.850553551899461
R-squared (Reduced Features): 0.4596685540565566

**PCA**
Mean Squared Error: 10.800929008597786
R-squared: 0.5075266684219414

- **R²** for Ridge Regression: ~0.43
- Further tuning of the models was performed to improve these results.

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

# Graphs

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/histogram_1.png) ![alt text](https://github.com/grant-i/Project_4/blob/main/figures/scatter_1.png)

# Shape of Data

```# Remove rows where RetailPrice is greater than 30
df_no_outliers = df[df['RetailPrice'] <= 30]
```

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/hist_2.png) ![alt text](https://github.com/grant-i/Project_4/blob/main/figures/scat_2.png)

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
# Random Forest Regresson

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


# Ridge

wait


# K Fold

```
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(linear_model, line_X, y, cv=kf, scoring='r2')
print(f"Mean cross-validated R-squared: {cv_scores.mean()}")
```

**Mean cross-validated R-squared: 0.35885767272885927**

Single R²: Might overestimate or underestimate performance based on one data split.

Mean Cross-validated R²: Provides a more balanced view of the model’s performance across different data partitions, which is typically more reliable for evaluating real-world performance.

![alt text](https://github.com/grant-i/Project_4/blob/main/figures/residual.png)


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
