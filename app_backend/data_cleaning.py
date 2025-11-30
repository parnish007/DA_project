import numpy as np
import pandas as pd
from helpers import safe_execute
from data_loader import df_copy, shape


class DataCleaning:

    @safe_execute
    def __init__(self):
        self.df_cpy = df_copy()

    # 1️⃣ Remove duplicates
    @safe_execute
    def remove_duplicate(self):
        self.df_cpy = self.df_cpy.drop_duplicates()
        return self.df_cpy

    # 2️⃣ Handle missing values
    @safe_execute
    def handle_missing_values(self, fill_value):
        self.df_cpy = self.df_cpy.fillna(value=fill_value)
        return self.df_cpy

    # 3️⃣ Drop specific columns
    @safe_execute
    def drop_columns(self, col_name):
        self.df_cpy = self.df_cpy.drop(columns=[col_name])
        return self.df_cpy

    # 4️⃣ Rename columns
    @safe_execute
    def rename_cols(self, prev_name, new_name):
        self.df_cpy = self.df_cpy.rename(columns={prev_name: new_name})
        return self.df_cpy
    
    # 5️⃣ Change datatype of a column
    @safe_execute
    def change_datatype(self, column_name, datatype):
        self.df_cpy[column_name] = self.df_cpy[column_name].astype(datatype)
        return self.df_cpy

    # 6️⃣ Reset index
    @safe_execute
    def reset_index(self):
        self.df_cpy = self.df_cpy.reset_index(drop=True)
        return self.df_cpy

    # 7️⃣ Apply ALL cleaning steps together (Pipeline)
    @safe_execute
    def get_cleaned_data(
        self,
        fill_value=None,
        drop_col=None,
        rename_pair=None,
        dtype_change=None
    ):
        """
        fill_value → value to fill NaN
        drop_col → column to drop
        rename_pair → tuple: ("old","new")
        dtype_change → tuple: ("col","dtype")
        """

        # Always remove duplicates first
        self.remove_duplicate()

        # Apply optional steps only if user passes values
        if fill_value is not None:
            self.handle_missing_values(fill_value)

        if drop_col is not None:
            self.drop_columns(drop_col)

        if rename_pair is not None:
            prev_name, new_name = rename_pair
            self.rename_cols(prev_name, new_name)

        if dtype_change is not None:
            col, typ = dtype_change
            self.change_datatype(col, typ)

        return self.df_cpy

    # 8️⃣ Show a preview of the current cleaned DataFrame
    @safe_execute
    def preview_change(self):
        return self.df_cpy.head()
