import pandas as pd
import numpy as np

file_path = "data/output_file.csv"

data = pd.read_csv(file_path)
data = data.iloc[:32952]

data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            
data = data.select_dtypes(include=[np.number])
cols_to_drop = ["customer_zip_code_prefix", "order_item_id", "product_name_lenght", "product_description_lenght", "product_photos_qty"]
data = data.drop(cols_to_drop, axis=1)
print(data.isna().sum())
# print(df.shape)
# data.to_csv("data/cleaned_data.csv", index=False)