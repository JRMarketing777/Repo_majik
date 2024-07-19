Here's a comprehensive cheatsheet for Pandas in markdown format:

# Pandas Cheatsheet

## Installation and Import

```python
pip install pandas
import pandas as pd
```

## Data Structures

### Series

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8])
s = pd.Series([1, 3, 5, 6, 8], index=['a', 'b', 'c', 'd', 'e'])
```

### DataFrame

```python
# From dictionary
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# From list of dictionaries
df = pd.DataFrame([{'A': 1, 'B': 2}, {'A': 3, 'B': 4}])

# From numpy array
df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))
```

## Reading/Writing Data

```python
# CSV
df = pd.read_csv('file.csv')
df.to_csv('file.csv', index=False)

# Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df.to_excel('file.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('file.json')
df.to_json('file.json')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table", conn)
df.to_sql('table_name', conn, if_exists='replace')
```

## Viewing Data

```python
df.head()  # First 5 rows
df.tail()  # Last 5 rows
df.info()  # Concise summary
df.describe()  # Statistical summary
df.shape  # (rows, columns)
df.columns  # Column labels
df.index  # Row labels
df.dtypes  # Data types of columns
```

## Selection

```python
# Selecting a single column
df['A']
df.A

# Selecting multiple columns
df[['A', 'B']]

# Selecting rows by label
df.loc[0]
df.loc[0:2]

# Selecting rows by integer position
df.iloc[0]
df.iloc[0:2]

# Boolean indexing
df[df['A'] > 0]
```

## Data Cleaning

```python
# Handling missing data
df.dropna()  # Drop rows with any NaN values
df.fillna(value=5)  # Fill NaN values with 5

# Removing duplicates
df.drop_duplicates()

# Renaming columns
df.rename(columns={'old_name': 'new_name'})

# Changing data types
df['A'] = df['A'].astype('int64')

# Replacing values
df.replace('old_value', 'new_value')
```

## Data Manipulation

```python
# Adding a new column
df['C'] = df['A'] + df['B']

# Applying a function to a column
df['A'].apply(lambda x: x*2)

# Sorting
df.sort_values(by='A', ascending=False)

# Grouping
df.groupby('A').sum()

# Merging
pd.merge(df1, df2, on='key_column')

# Concatenating
pd.concat([df1, df2])

# Pivoting
df.pivot(index='A', columns='B', values='C')

# Melting
pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
```

## Time Series

```python
# Creating date range
dates = pd.date_range('20230101', periods=6)

# Resampling
df.resample('D').mean()  # Daily mean
df.resample('M').sum()   # Monthly sum

# Time zone conversion
df.tz_localize('UTC').tz_convert('US/Eastern')

# Rolling statistics
df.rolling(window=7).mean()  # 7-day rolling average
```

## Statistical Methods

```python
df.mean()
df.median()
df.mode()
df.min()
df.max()
df.std()
df.var()
df.corr()  # Correlation matrix
df.cov()   # Covariance matrix
```

## Plotting

```python
import matplotlib.pyplot as plt

df.plot()  # Line plot
df.plot.bar()  # Bar plot
df.plot.hist()  # Histogram
df.plot.box()  # Box plot
df.plot.scatter(x='A', y='B')  # Scatter plot

plt.show()
```

## Advanced Features

```python
# String methods
df['A'].str.lower()
df['A'].str.contains('pattern')

# Categorical data
df['category'] = pd.Categorical(df['category'])

# Multi-index
df = df.set_index(['A', 'B'])

# Cross-tabulation
pd.crosstab(df['A'], df['B'])

# Binning
pd.cut(df['A'], bins=3)
```

## Performance and Memory Usage

```python
# Using appropriate dtypes
df['int_col'] = df['int_col'].astype('int32')
df['float_col'] = df['float_col'].astype('float32')

# Using categories for string columns with few unique values
df['category_col'] = df['category_col'].astype('category')

# Checking memory usage
df.info(memory_usage='deep')

# Optimizing with chunksize when reading large files
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

This cheatsheet covers the most commonly used Pandas operations and functions. Remember to refer to the official Pandas documentation for more detailed information and advanced usage.
