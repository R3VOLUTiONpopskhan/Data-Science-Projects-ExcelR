"""Run from the project folder: python src/analysis.py."""

# # Sales and Discounts: Descriptive Statistics and Preprocessing
# 
# This project examines 450 sales records supplied with the ExcelR Basic Statistics assignment. It covers every requested numerical summary, histograms, boxplots, category counts, z-score scaling, and one-hot encoding. Monetary values are reported in the dataset's original units because no currency is specified.
# 
# The analysis preserves the original CSV. Potential outliers are flagged for review, not automatically deleted. This is exploratory analysis and preprocessing, not a predictive model.


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display

BASE = Path.cwd()
if BASE.name in ('notebooks', 'src'):
    BASE = BASE.parent
for folder in ('results', 'figures'):
    (BASE / folder).mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})

# ## 1. Load and inspect the data
# Dates are parsed as day-month-year. Date is a time variable, not a nominal category. SKU and Model remain categorical identifiers.


df = pd.read_csv(BASE / 'data/sales_data_with_discounts.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='raise')
numeric = df.select_dtypes(include='number').columns.tolist()
categorical = [c for c in df.columns if c not in numeric + ['Date']]
print(f'Shape: {df.shape}; date range: {df.Date.min().date()} to {df.Date.max().date()}')
print('Numeric columns:', numeric)
print('Categorical columns:', categorical)
display(df.head())
quality = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(),
                        'unique': df.nunique()})
display(quality)
print('Duplicate complete rows:', df.duplicated().sum())
print('Duplicate date/SKU combinations:', df.duplicated(['Date', 'SKU']).sum())
assert not df.isna().any().any(), 'Review missing values before preprocessing.'
assert (df['Day'] == df['Date'].dt.day_name()).all()
assert np.allclose(df['Total Sales Value'], df.Volume * df['Avg Price'])
assert np.allclose(df['Discount Amount'], df['Total Sales Value'] * df['Discount Rate (%)'] / 100)
assert np.allclose(df['Net Sales Value'], df['Total Sales Value'] - df['Discount Amount'])
print('All three sales arithmetic checks passed.')
quality.to_csv(BASE / 'results/data_quality.csv')

# ## 2. Descriptive statistics
# The mean measures the arithmetic average, while the median is less sensitive to extreme observations. Standard deviation uses the **sample convention, ddof=1**. All tied modes are retained. A continuous column with every value unique has no informative mode, rather than a meaningful list of 450 modes.


summary = df[numeric].describe().T
summary['median'] = df[numeric].median()
summary['skewness'] = df[numeric].skew()
def modes_text(s):
    counts = s.value_counts()
    if counts.max() == 1:
        return 'No repeated value'
    return ', '.join(f'{v:g}' for v in sorted(counts[counts == counts.max()].index))
summary['mode'] = [modes_text(df[c]) for c in numeric]
display(summary[['mean', 'median', 'mode', 'std', 'min', '25%', '75%', 'max', 'skewness']])
summary.to_csv(BASE / 'results/descriptive_statistics.csv')
for c in numeric:
    r = summary.loc[c]
    print(f'{c}: mean={r["mean"]:.3f}, median={r["median"]:.3f}, '
          f'sample SD={r["std"]:.3f}, skewness={r["skewness"]:.3f}.')

# ## 3. Histograms and boxplots
# Separate axes prevent the different measurement scales from obscuring the distributions. Histogram bin choice affects the visual appearance. The boxplot uses the 1.5 × IQR rule: observations below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are flagged. This rule identifies unusual values; it does not prove that they are errors.


fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for ax, c in zip(axes.flat, numeric):
    ax.hist(df[c], bins=25, color='#266b99', edgecolor='white')
    ax.set(title=c, xlabel='Value', ylabel='Record count')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
fig.suptitle('Numerical distributions')
fig.savefig(BASE / 'figures/histograms.png')
plt.show()
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
outlier_rows = []
for ax, c in zip(axes.flat, numeric):
    ax.boxplot(df[c], orientation='horizontal', patch_artist=True,
               boxprops={'facecolor': '#a4cee5'})
    ax.set(title=c, xlabel='Value', yticks=[])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    mask = (df[c] < lower) | (df[c] > upper)
    outlier_rows.append([c, q1, q3, iqr, lower, upper, int(mask.sum())])
fig.suptitle('Boxplots: possible outliers retained')
fig.savefig(BASE / 'figures/boxplots.png')
plt.show()
outliers = pd.DataFrame(outlier_rows, columns=['variable','Q1','Q3','IQR','lower_fence','upper_fence','flagged_count'])
display(outliers)
for row in outliers.itertuples(index=False):
    print(f'{row.variable}: IQR={row.IQR:.3f}; {row.flagged_count} of {len(df)} records outside the fences.')
outliers.to_csv(BASE / 'results/outlier_summary.csv', index=False)

# ### Interpretation
# Volume, average price, total sales, discount amount, and net sales have positive skew: their means exceed their medians, and a relatively small number of large observations pull the averages upward. Discount rate is negatively skewed, with most observations toward the upper end of its range. Price and sales distributions combine different product groups, so overall summaries hide group differences. The flagged values should be checked against products and transaction context before any removal.

# ## 4. Category frequencies and date coverage
# Frequency means the number of records, not units sold or revenue. Every category is shown, including the 30 SKU and Model labels. Dates are visualized separately as chronological coverage.


category_rows = []
for c in categorical:
    counts = df[c].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(3, len(counts)*.27)), constrained_layout=True)
    counts.plot.barh(ax=ax, color='#266b99')
    ax.set(title=f'{c}: record frequency', xlabel='Record count', ylabel=c)
    fig.savefig(BASE / 'figures' / f'category_{c.lower()}.png')
    plt.show()
    for value, count in counts.items():
        category_rows.append([c, value, count, count/len(df)])
pd.DataFrame(category_rows, columns=['variable','category','count','share']).to_csv(
    BASE/'results/category_frequencies.csv', index=False)
daily = df.groupby('Date').size()
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.bar(daily.index.strftime('%d %b'), daily.values, color='#266b99')
ax.tick_params(axis='x', rotation=60)
ax.set(title='Records per date', ylabel='Record count')
fig.savefig(BASE/'figures/date_coverage.png'); plt.show()
display(df.groupby('BU').agg(records=('SKU','size'), units=('Volume','sum'),
                             net_sales=('Net Sales Value','sum')))
print('Brand counts:'); print(df.Brand.value_counts().to_string())

# ### Interpretation
# The dataset covers 15 consecutive dates with 30 records per date. Each of the three business units has 150 records; each SKU and Model occurs 15 times. City is constant (C), so it provides no discrimination in this sample. Jeera is the most frequent brand with 90 records (20%). Weekday counts reflect calendar coverage: Thursday has 90 records, while each other weekday has 60. Equal record counts do not imply equal sales or demand. The regular coverage suggests a structured sample whose representativeness of wider sales is unknown.

# ## 5. Standardize numerical variables
# For each numerical column, **z = (x − μ) / σ**. Here μ and σ are the mean and population-convention standard deviation of the available dataset (**ddof=0**), matching common machine-learning scalers. This differs intentionally from the sample SD used in the descriptive table.
# 
# Standardization changes units and scale, but preserves shape, skewness, and relative positions. It does not make a skewed distribution normal or remove outliers. For predictive modeling, split the data first and estimate scaling parameters on the training set only.


means = df[numeric].mean()
scales = df[numeric].std(ddof=0)
assert (scales > 0).all(), 'Constant numeric columns need separate handling.'
standardized = (df[numeric] - means) / scales
comparison = pd.DataFrame({'before_mean': means, 'before_std_ddof0': scales,
                          'after_mean': standardized.mean(),
                          'after_std_ddof0': standardized.std(ddof=0)})
display(comparison)
display(standardized.head())
assert np.allclose(standardized.mean(), 0, atol=1e-12)
assert np.allclose(standardized.std(ddof=0), 1, atol=1e-12)
standardized.to_csv(BASE/'results/standardized_numeric.csv', index=False)
comparison.to_csv(BASE/'results/scaling_parameters_and_checks.csv')
fig, axes = plt.subplots(len(numeric), 2, figsize=(12, 16), constrained_layout=True)
for row, c in enumerate(numeric):
    axes[row,0].hist(df[c], bins=25, color='#266b99', edgecolor='white')
    axes[row,0].set(title=f'{c}: original', ylabel='Record count', xlabel='Original value')
    axes[row,1].hist(standardized[c], bins=25, color='#22866c', edgecolor='white')
    axes[row,1].set(title=f'{c}: standardized', xlabel='z-score', ylabel='Record count')
    for ax in axes[row]:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
fig.savefig(BASE/'figures/before_after_standardization.png')
plt.show()

# ## 6. One-hot encode categorical variables
# Nominal categories have no inherent numerical order. One-hot encoding creates a binary indicator for each category without imposing an artificial ranking. All levels are retained to show the full assignment transformation; a linear model with an intercept generally needs a reference level removed or another method to handle collinearity.
# 
# Date remains in the raw data and is excluded from the feature matrix. Real modeling would derive appropriate calendar features. City is constant and SKU and Model are redundant identifiers; all are included for this demonstration but should be reconsidered for a particular modeling goal.


dummies = pd.get_dummies(df[categorical], columns=categorical, dtype=int, drop_first=False)
assert set(np.unique(dummies.to_numpy())) <= {0, 1}
assert (dummies.sum(axis=1) == len(categorical)).all()
transformed = pd.concat([standardized, dummies], axis=1)
assert transformed.shape[0] == df.shape[0]
assert not transformed.isna().any().any()
print(f'{len(categorical)} categorical variables -> {dummies.shape[1]} binary columns')
print('Final feature matrix:', transformed.shape)
display(transformed.iloc[:5, :16])
transformed.to_csv(BASE/'results/preprocessed_features.csv', index=False)
mapping = {c: sorted(df[c].unique().tolist()) for c in categorical}
import json
(BASE/'results/category_levels.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')

# ## 7. Conclusion
# Most sales measures are strongly right-skewed, so median and IQR complement mean and standard deviation. The fixed date/product coverage and constant city limit generalization. Outlier flags are retained for investigation. Scaling places six numerical features on a comparable scale; one-hot encoding represents six categorical variables as 80 binary indicators, giving 86 numeric features.
# 
# These transformations demonstrate preprocessing, but they are not a ready-to-train prediction pipeline. A future model needs a defined target, an appropriate split (often time-based), and preprocessing fitted only on training data. Related accounting variables can leak a target: for example, total sales and discount amount determine net sales exactly. No causal or predictive claim is made here.
# 
# **Source:** supplied `Basic stats - 1.zip`, containing `Basic statistics.docx` and the original CSV. Assignment requirements are summarized here; the raw CSV is unchanged.
