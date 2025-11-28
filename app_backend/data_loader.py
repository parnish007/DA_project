
import pandas as pd

from helpers import safe_execute


class data_loader:


    @safe_execute
    def __init__(self,dataset):
        self.df=pd.read_csv(dataset)
        print(f"this is your data:{self.df.head(4)}")

  
    @safe_execute
    def get_colums(self):
        self.columns=self.df.columns
        return self.columns
    @safe_execute
    def shape(self):
        self.shape= self.df.shape()
        return self.shape
    @safe_execute
    def preview_by_index(self,n):
        self.rows_by_indx=self.df.iloc[n]
        return self.rows_by_indx
    @safe_execute
    def preview_by_val(self,val):
        self.rows_by_value=self.df.loc[val]
        return self.rows_by_value
    @safe_execute
    def df_copy(self):
        self.copy_df=self.df.copy()
        return self.copy_df


data1= data_loader(path)
