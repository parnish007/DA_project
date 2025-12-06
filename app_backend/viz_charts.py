# viz_charts.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.express as px
import os

class VizEngine:
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty.")
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    def histogram(self, column: str, bins: int = 30, kde: bool = False):
        if column not in self.df.columns:
            raise KeyError(f"{column} not in DataFrame")
        if not np.issubdtype(self.df[column].dtype, np.number):
            raise TypeError(f"{column} is not numeric")
        
        data = self.df[column].dropna()
        fig, ax = plt.subplots()
        counts, bins, patches = ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        
        kde_x, kde_y = None, None
        if kde:
            kde_func = gaussian_kde(data)
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde_y = kde_func(kde_x)
            ax.plot(kde_x, kde_y * len(data) * (bins[1]-bins[0]), color='red')
        
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        return fig if not kde else {"bins": bins.tolist(), "counts": counts.tolist(), "kde_x": kde_x.tolist(), "kde_y": kde_y.tolist()}

    def boxplot(self, column: str):
        if column not in self.df.columns:
            raise KeyError(f"{column} not in DataFrame")
        if not np.issubdtype(self.df[column].dtype, np.number):
            raise TypeError(f"{column} is not numeric")
        
        data = self.df[column].dropna()
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = data[(data < lower_fence) | (data > upper_fence)]
        
        fig, ax = plt.subplots()
        ax.boxplot(data, vert=True, patch_artist=True)
        ax.set_title(f'Boxplot of {column}')
        return fig, {
            "q1": q1, "median": median, "q3": q3,
            "iqr": iqr, "lower_fence": lower_fence,
            "upper_fence": upper_fence, "outliers": outliers.tolist()
        }

    def bar_chart(self, column: str, top_n: int = 20):
        if column not in self.df.columns:
            raise KeyError(f"{column} not in DataFrame")
        
        counts = self.df[column].value_counts().head(top_n)
        others_count = self.df[column].value_counts()[top_n:].sum()
        if others_count > 0:
            counts["Others"] = others_count
        
        fig, ax = plt.subplots()
        counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Bar Chart of {column}')
        ax.set_ylabel('Count')
        ax.set_xlabel(column)
        return fig, {"labels": counts.index.tolist(), "values": counts.values.tolist()}

    def scatter(self, x_col: str, y_col: str, sample_fraction: float = 0.15):
        for col in [x_col, y_col]:
            if col not in self.df.columns:
                raise KeyError(f"{col} not in DataFrame")
            if not np.issubdtype(self.df[col].dtype, np.number):
                raise TypeError(f"{col} is not numeric")
        
        df_sample = self.df[[x_col, y_col]].dropna()
        if sample_fraction < 1.0:
            df_sample = df_sample.sample(frac=sample_fraction, random_state=42)
        
        fig, ax = plt.subplots()
        ax.scatter(df_sample[x_col], df_sample[y_col], alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        return fig, {"x": df_sample[x_col].tolist(), "y": df_sample[y_col].tolist(), "n_points": len(df_sample)}

    def pairplot(self, cols=None, sample_fraction=0.1, max_plots=6):
        cols = cols or self.numeric_cols[:max_plots]
        if len(cols) > max_plots:
            cols = cols[:max_plots]
        df_sample = self.df[cols].dropna()
        if sample_fraction < 1.0:
            df_sample = df_sample.sample(frac=sample_fraction, random_state=42)
        
        sns.set(style="ticks")
        fig = sns.pairplot(df_sample)
        return fig

    def heatmap(self, corr_matrix: pd.DataFrame, annot=False):
        fig, ax = plt.subplots(figsize=(max(6, len(corr_matrix)/2), max(4, len(corr_matrix)/3)))
        sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        return fig

    def save_plot(self, fig, path: str, dpi: int = 300):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return os.path.abspath(path)
