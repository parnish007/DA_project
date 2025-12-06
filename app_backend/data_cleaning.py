import pandas as pd
import numpy as np
from helpers import safe_execute
from data_loader import df_copy


class DataCleaning:

    @safe_execute
    def __init__(self):
        self.df = df_copy()   # copy fresh dataset

   
    # 1️⃣ REMOVE DUPLICATES
   
    @safe_execute
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self.df

   
    # 2️⃣ HANDLE MISSING VALUES 
   
    @safe_execute
    def handle_missing_values(self, strategy="drop", custom_value=None):
        """
        strategy options:
        - drop
        - fill_mean
        - fill_median
        - fill_mode
        - fill_custom
        """

        if strategy == "drop":
            self.df = self.df.dropna()

        elif strategy == "fill_mean":
            mean_vals = self.df.mean(numeric_only=True)
            self.df = self.df.fillna(mean_vals)

        elif strategy == "fill_median":
            median_vals = self.df.median(numeric_only=True)
            self.df = self.df.fillna(median_vals)

        elif strategy == "fill_mode":
            mode_vals = self.df.mode().iloc[0]
            self.df = self.df.fillna(mode_vals)

        elif strategy == "fill_custom":
            if custom_value is None:
                raise ValueError("Pass custom_value for fill_custom strategy")
            self.df = self.df.fillna(custom_value)

        else:
            raise ValueError("Invalid strategy")

        return self.df

   
    # 3️⃣ DROP COLUMNS
   
    @safe_execute
    def drop_columns(self, col_list):
        if type(col_list) not in [list, tuple]:
            raise ValueError("col_list must be a list")
        self.df = self.df.drop(columns=col_list, errors="ignore")
        return self.df

   
    # 4️⃣ RENAME COLUMNS
   
    @safe_execute
    def rename_columns(self, rename_dict):
        if not isinstance(rename_dict, dict):
            raise ValueError("rename_dict must be a dictionary")
        self.df = self.df.rename(columns=rename_dict)
        return self.df

   
    # 5️⃣ CHANGE DATA TYPES
   
    @safe_execute
    def change_datatypes(self, dtype_dict):
        """
        dtype_dict example:
        {"age": "int", "salary": "float"}
        """
        if not isinstance(dtype_dict, dict):
            raise ValueError("dtype_dict must be a dictionary")

        for col, dtype in dtype_dict.items():
            self.df[col] = self.df[col].astype(dtype)

        return self.df

   
    # 6️⃣ SIMPLE OUTLIER HANDLING 
   
    @safe_execute
    def handle_outliers(self, columns=None):
        """
        Removes rows where values are extremely high or low
        using the simple IQR method (Beginner-friendly)
        """

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Keep only values within limits
            self.df = self.df[(self.df[col] >= lower_limit) & (self.df[col] <= upper_limit)]

        return self.df

   
    # 7️⃣ RESET INDEX
   
    @safe_execute
    def reset_index(self):
        self.df = self.df.reset_index(drop=True)
        return self.df

   
    # 8️⃣ GET CLEANED DATA (SAFE COPY)
   
    @safe_execute
    def get_cleaned_data(self):
        return self.df.copy()

   
    # 9️⃣ PREVIEW CHANGES
   
    @safe_execute
    def preview_changes(self, n=5):
        return self.df.head(n)
