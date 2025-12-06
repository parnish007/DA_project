import pandas as pd
from .helpers import safe_execute

class DataLoader:
    def __init__(self, dataset):
        """Load dataset from path or BytesIO."""
        try:
            self.df = pd.read_csv(dataset)
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")

    @safe_execute()
    def df_copy(self):
        return self.df.copy()

    @safe_execute()
    def get_columns(self):
        return self.df.columns.tolist()

    @safe_execute()
    def shape(self):
        return self.df.shape

    @safe_execute()
    def preview_by_index(self, n):
        return self.df.iloc[n]

    @safe_execute()
    def preview_by_val(self, val):
        return self.df.loc[val]
