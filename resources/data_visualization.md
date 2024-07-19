# Data Visualization Libraries Cheatsheet

## Matplotlib

### Description
Matplotlib is a widely used Python library for creating static, animated, and interactive visualizations. It provides a flexible and powerful interface for generating a variety of plots.

### Installation

```bash
pip install matplotlib
```

### Basic Usage

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple line plot
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()
```

### Common Plot Types

- **Line Plot:**
  ```python
  plt.plot(x, y)
  ```

- **Scatter Plot:**
  ```python
  plt.scatter(x, y)
  ```

- **Bar Plot:**
  ```python
  plt.bar(['A', 'B', 'C'], [3, 7, 5])
  ```

- **Histogram:**
  ```python
  plt.hist(data, bins=10)
  ```

### Use Cases
- Creating publication-quality plots.
- Visualizing data distributions and trends.
- Generating custom plots for data analysis.

---

## Seaborn

### Description
Seaborn is a statistical data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

### Installation

```bash
pip install seaborn
```

### Basic Usage

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
tips = sns.load_dataset('tips')

# Create a scatter plot with regression line
sns.regplot(x='total_bill', y='tip', data=tips)
plt.title('Total Bill vs Tip')
plt.show()
```

### Common Plot Types

- **Scatter Plot:**
  ```python
  sns.scatterplot(x='total_bill', y='tip', data=tips)
  ```

- **Box Plot:**
  ```python
  sns.boxplot(x='day', y='total_bill', data=tips)
  ```

- **Heatmap:**
  ```python
  sns.heatmap(data.corr(), annot=True)
  ```

- **Pair Plot:**
  ```python
  sns.pairplot(tips)
  ```

### Use Cases
- Visualizing relationships between multiple variables.
- Creating attractive and informative statistical graphics.
- Exploring datasets with built-in themes and color palettes.

---

## Plotly

### Description
Plotly is a library for creating interactive plots and dashboards. It supports a wide range of chart types and is particularly useful for web-based visualizations.

### Installation

```bash
pip install plotly
```

### Basic Usage

```python
import plotly.express as px

# Load example dataset
df = px.data.iris()

# Create an interactive scatter plot
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species', title='Iris Dataset')
fig.show()
```

### Common Plot Types

- **Scatter Plot:**
  ```python
  fig = px.scatter(df, x='total_bill', y='tip', color='day')
  ```

- **Bar Plot:**
  ```python
  fig = px.bar(df, x='day', y='total_bill', color='sex')
  ```

- **Line Plot:**
  ```python
  fig = px.line(df, x='date', y='value', title='Time Series Data')
  ```

- **3D Scatter Plot:**
  ```python
  fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length', color='species')
  ```

### Use Cases
- Creating interactive dashboards for data exploration.
- Visualizing complex datasets with multiple dimensions.
- Sharing visualizations in web applications and reports.

---

### Summary

- **Matplotlib** is the foundation for many plotting tasks, providing extensive customization options for static and animated plots.
- **Seaborn** simplifies the creation of statistical graphics, making it easier to visualize complex datasets with attractive defaults.
- **Plotly** excels in creating interactive visualizations, making it suitable for web applications and dashboards.

These libraries can be used individually or in combination to create compelling visual representations of your data, facilitating better insights and communication of findings.
