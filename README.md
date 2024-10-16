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
    - **Ridge Regression**
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

### Feature Details:
- **g_sum**: Represents the rounded sum of the fat, carbohydrates, proteins, and salt values in the data. This helps identify products with potential false entries.
- **reconstructed_energy**: Calculates the energy value of a product based on fat, carbohydrates, and proteins. It is compared with the dataset's given energy values to check for discrepancies.

These features contribute to the understanding of a food product's nutritional profile and play a crucial role in predictive modeling.

## Performance Metrics

The performance of the models was evaluated using metrics such as R² and Mean Squared Error (MSE):

- **R²** for Ridge Regression: ~0.43
- Further tuning of the models was performed to improve these results.

## Installation & Dependencies

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/grant-i/Project_4

2. Install required dependencies:
   You can create a `requirements.txt` file and include the following packages to make installation easier:
   ```txt
   pandas
   scikit-learn
   seaborn
   matplotlib
   numpy
   hvplot
   plotly
   statsmodels

## Data Loading, Standardization, and Cleaning

### Data Format 

1. Loading the Data

Here's an example of loading the datasets used in the project:

```python
def load_data(file_path):
    path = Path(file_path)
    if path.is_file():
        return pd.read_csv(path, encoding='ISO-8859-1')
    else:
        print(f"Error: {file_path} not found.")
        return pd.DataFrame()

``` Load datasets
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