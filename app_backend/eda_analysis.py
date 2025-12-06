# eda_analysis.py
import math
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# optional dependency used only if present — kept guarded
try:
    from scipy.stats import zscore, gaussian_kde, skew, kurtosis
except Exception:
    zscore = None
    gaussian_kde = None
    def skew(arr): return float(pd.Series(arr).skew())
    def kurtosis(arr): return float(pd.Series(arr).kurtosis())


class EDAEngine:
    """
    Beginner-friendly EDA engine.
    - Does NOT mutate the original DataFrame provided at init.
    - Returns pandas objects and basic Python structures easily consumable by frontends.
    """

    def __init__(self, df: pd.DataFrame, sample_limit: Optional[int] = None):
        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError("df must be a non-empty pandas DataFrame")
        self.df = df.copy()
        self.sample_limit = int(sample_limit) if sample_limit is not None else None

        self.numeric_cols = self._get_numeric_cols()
        self.categorical_cols = self._get_categorical_cols()
        self.summary_cache: Dict[str, Any] = {}

    # -------------------------
    # helpers
    # -------------------------
    def _sample_df(self) -> pd.DataFrame:
        if self.sample_limit is None or len(self.df) <= self.sample_limit:
            return self.df
        return self.df.sample(n=self.sample_limit, random_state=42)

    def _get_numeric_cols(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_categorical_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    def _validate_column(self, column: str):
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

    # -------------------------
    # 1. Overview
    # -------------------------
    def get_overview(self, top_n_cols: Optional[int] = 10) -> dict:
        d = self.df
        rows, cols = d.shape
        num_missing = int(d.isna().sum().sum())
        numeric = len(self._get_numeric_cols())
        categorical = len(self._get_categorical_cols())

        try:
            mem_bytes = d.memory_usage(deep=True).sum()
            memory_mb = float(mem_bytes) / (1024 ** 2)
        except Exception:
            memory_mb = float("nan")

        top_n_cols_list = None
        if top_n_cols:
            non_null_counts = d.notna().sum().sort_values(ascending=False)
            top_n_cols_list = non_null_counts.index[:top_n_cols].tolist()

        return {
            "rows": int(rows),
            "columns": int(cols),
            "num_numeric": int(numeric),
            "num_categorical": int(categorical),
            "num_missing": int(num_missing),
            "memory_usage_mb": round(memory_mb, 3) if not math.isnan(memory_mb) else None,
            "top_n_cols": top_n_cols_list,
        }

    # -------------------------
    # 2. Summary stats
    # -------------------------
    def summary_stats(self, columns: Optional[List[str]] = None,
                      include_percentiles: List[float] = [0.25, 0.5, 0.75]) -> pd.DataFrame:
        if columns is None:
            columns = self._get_numeric_cols()
        else:
            for c in columns:
                self._validate_column(c)

        if not columns:
            return pd.DataFrame({"Info": ["No numeric columns to summarize"]})

        sample = self._sample_df()
        if sample.empty:
            return pd.DataFrame({"Info": ["No data available for summary"]})

        stats = sample[columns].describe(percentiles=include_percentiles).T
        stats = stats.rename(columns={"50%": "median"}) if "50%" in stats.columns else stats
        stats["dtype"] = sample[columns].dtypes.astype(str).values
        return stats

    # -------------------------
    # 3. Value counts
    # -------------------------
    def value_counts(self, column: str, top_n: int = 20, normalize: bool = False) -> pd.DataFrame:
        self._validate_column(column)
        series = self.df[column].value_counts(normalize=normalize).head(top_n)
        result = series.reset_index()
        result.columns = ["value", "proportion" if normalize else "count"]
        return result

    # -------------------------
    # 4. Missing data report
    # -------------------------
    def missing_data_report(self, sort_by: str = "missing_pct", threshold: float = 0.0) -> pd.DataFrame:
        if sort_by not in {"missing_pct", "missing_count"}:
            raise ValueError("sort_by must be 'missing_pct' or 'missing_count'")

        total = len(self.df)
        report = pd.DataFrame({
            "column": self.df.columns,
            "missing_count": self.df.isna().sum().astype(int),
            "missing_pct": (self.df.isna().sum() / total).astype(float),
            "dtype": self.df.dtypes.astype(str),
            "unique_count": self.df.nunique(dropna=False).astype(int)
        }).reset_index(drop=True)

        if threshold > 0:
            report = report[report["missing_pct"] >= float(threshold)]

        report = report.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        return report

    # -------------------------
    # 5. Correlation matrix
    # -------------------------
    def correlation_matrix(self, method: str = "pearson", numeric_only: bool = True,
                           clamp_threshold: Optional[float] = None) -> pd.DataFrame:
        if numeric_only:
            cols = self._get_numeric_cols()
            if not cols:
                return pd.DataFrame({"Info": ["No numeric columns for correlation"]})
            sample = self._sample_df()
            if sample.empty:
                return pd.DataFrame({"Info": ["No data available for correlation"]})
            corr = sample[cols].corr(method=method)
        else:
            cols = self.df.columns.tolist()
            temp = self.df.copy()
            for c in cols:
                if not pd.api.types.is_numeric_dtype(temp[c]):
                    temp[c] = pd.Categorical(temp[c]).codes
            corr = temp[cols].corr(method=method)

        if clamp_threshold is not None:
            corr = corr.where(corr.abs() >= clamp_threshold, other=0.0)
        return corr

    # -------------------------
    # 6. Top correlations
    # -------------------------
    def top_correlations(self, target_col: Optional[str] = None, n: int = 10, method: str = "pearson") -> pd.DataFrame:
        corr = self.correlation_matrix(method=method, numeric_only=True)
        if "Info" in corr.columns:
            return corr
        if target_col:
            self._validate_column(target_col)
            if target_col not in corr.columns:
                raise KeyError(f"'{target_col}' is not numeric or not present in correlation matrix")
            series = corr[target_col].drop(labels=[target_col], errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
            result = series.head(n).reset_index()
            result.columns = ["feature", "corr"]
            return result
        else:
            m = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False)
            top = m.head(n)
            pairs = top.reset_index()
            pairs.columns = ["feature_a", "feature_b", "abs_corr"]
            pairs["corr"] = pairs.apply(lambda r: corr.loc[r["feature_a"], r["feature_b"]], axis=1)
            return pairs[["feature_a", "feature_b", "corr"]]

    # -------------------------
    # 7. Distribution stats
    # -------------------------
    def distribution_stats(self, column: str, bins: int = 30, kde: bool = False) -> dict:
        self._validate_column(column)
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"Info": "distribution_stats requires a numeric column"}
        series = self._sample_df()[column].dropna()
        if series.empty:
            return {"Info": "No data available for distribution"}

        hist_counts, hist_bins = np.histogram(series, bins=bins)
        out = {
            "hist_counts": hist_counts.tolist(),
            "hist_bins": hist_bins.tolist(),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out["outlier_counts"] = int(((series < lower) | (series > upper)).sum())

        if kde:
            if gaussian_kde is None:
                out["kde_error"] = "scipy not available for KDE"
            else:
                try:
                    kde_model = gaussian_kde(series)
                    xs = np.linspace(series.min(), series.max(), 200)
                    ys = kde_model(xs)
                    out["kde_x"] = xs.tolist()
                    out["kde_y"] = ys.tolist()
                except Exception as e:
                    out["kde_error"] = str(e)
        return out

    # -------------------------
    # 8. Boxplot summary
    # -------------------------
    def boxplot_summary(self, column: str) -> dict:
        self._validate_column(column)
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"Info": "boxplot_summary requires a numeric column"}
        series = self.df[column].dropna()
        if series.empty:
            return {"Info": "No data available for boxplot"}

        q1 = float(series.quantile(0.25))
        median = float(series.quantile(0.5))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers_mask = (series < lower_fence) | (series > upper_fence)
        outlier_indices = series.index[outliers_mask].tolist()
        outlier_values = series.loc[outlier_indices].tolist()

        return {
            "min": float(series.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(series.max()),
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values
        }

    # -------------------------
    # 9. Pairwise scatter
    # -------------------------
    def pairwise_scatter(self, sample_fraction: float = 0.1, cols: Optional[List[str]] = None, max_plots: int = 9) -> List[dict]:
        if cols is None:
            numeric = self._get_numeric_cols()
            variances = self.df[numeric].var().sort_values(ascending=False) if numeric else pd.Series()
            cols = variances.index.tolist()[:min(len(variances), 6)] if not variances.empty else []
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j]))
                if len(pairs) >= max_plots:
                    break
            if len(pairs) >= max_plots:
                break
        n = max(1, int(len(self.df) * sample_fraction))
        sample_df = self.df.sample(n=n, random_state=42) if n < len(self.df) else self.df
        out = []
        for x_col, y_col in pairs:
            out.append({
                "x_col": x_col,
                "y_col": y_col,
                "x_values": sample_df[x_col].dropna().tolist(),
                "y_values": sample_df[y_col].dropna().tolist()
            })
        return out

    # -------------------------
    # 10. Categorical summary
    # -------------------------
    def categorical_summary(self, cols: Optional[List[str]] = None, top_n: int = 10) -> pd.DataFrame:
        if cols is None:
            cols = self._get_categorical_cols()
        rows = []
        for c in cols:
            values = self.df[c].value_counts(dropna=False).head(top_n)
            top_values = list(zip(values.index.tolist(), values.tolist()))
            rows.append({
                "column": c,
                "dtype": str(self.df[c].dtype),
                "unique_count": int(self.df[c].nunique(dropna=False)),
                "top_values": top_values
            })
        return pd.DataFrame(rows)

    # -------------------------
    # 11. Time series overview
    # -------------------------
    def time_series_overview(self, datetime_col: str, value_cols: Optional[List[str]] = None,
                             freq: str = 'D', agg: str = 'mean') -> pd.DataFrame:
        self._validate_column(datetime_col)
        ser = pd.to_datetime(self.df[datetime_col], errors='coerce')
        if ser.isna().all():
            return pd.DataFrame({"Info": [f"Column '{datetime_col}' cannot be parsed as datetime"]})
        temp = self.df.copy()
        temp[datetime_col] = ser
        temp = temp.set_index(datetime_col)
        if value_cols is None:
            value_cols = self._get_numeric_cols()
        if not value_cols:
            return pd.DataFrame({"Info": ["No numeric columns for time series aggregation"]})
        result = temp[value_cols].resample(freq).agg(agg)
        return result

    # -------------------------
    # 12. Groupby aggregations
    # -------------------------
    def groupby_aggregations(self, group_cols: List[str], agg_spec: dict) -> pd.DataFrame:
        for g in group_cols:
            self._validate_column(g)
        result = self.df.groupby(group_cols).agg(agg_spec).reset_index()
        return result

    # -------------------------
    # 13. Feature types report
    # -------------------------
    def feature_types_report(self) -> pd.DataFrame:
        rows = []
        for c in self.df.columns:
            dtype = str(self.df[c].dtype)
            sample_values = self.df[c].dropna().unique()[:5].tolist()
            if pd.api.types.is_numeric_dtype(self.df[c]):
                suggested = "numeric"
            elif pd.api.types.is_bool_dtype(self.df[c]):
                suggested = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(self.df[c]) or "datetime" in dtype:
                suggested = "datetime"
            elif pd.api.types.is_categorical_dtype(self.df[c]) or pd.api.types.is_object_dtype(self.df[c]):
                uniq = self.df[c].nunique(dropna=True)
                suggested = "categorical (high_cardinality)" if uniq > 50 else "categorical"
            else:
                suggested = "other"
            rows.append({
                "column": c,
                "dtype": dtype,
                "sample_values": sample_values,
                "unique_count": int(self.df[c].nunique(dropna=False)),
                "recommended_action": suggested
            })
        return pd.DataFrame(rows)

    # -------------------------
    # 14. Export report
    # -------------------------
    def export_report(self, format: str = 'csv', path: Optional[str] = None, include_plots: bool = False) -> Any:
        fmt = format.lower()
        if fmt not in {'csv', 'json', 'html'}:
            raise ValueError("format must be one of 'csv','json','html'")
        cleaned = self.df.copy()
        if path:
            if fmt == 'csv':
                cleaned.to_csv(path, index=False)
            elif fmt == 'json':
                cleaned.to_json(path, orient='records', date_format='iso')
            elif fmt == 'html':
                cleaned.to_html(path, index=False)
            return path
        if fmt == 'csv':
            return cleaned.to_csv(index=False).encode('utf-8')
        elif fmt == 'json':
            return cleaned.to_json(orient='records', date_format='iso')
        else:
            return cleaned.to_html(index=False)
