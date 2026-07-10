print("Program Started")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("iris.csv")

# Display first rows
print("First 5 Rows")
print(df.head())

# Dataset Info
print("\nDataset Info")
print(df.info())

# Shape
print("\nShape")
print(df.shape)

# Summary Statistics
print("\nSummary Statistics")
print(df.describe())

# Mean
print("\nMean")
print(df.mean(numeric_only=True))

# Median
print("\nMedian")
print(df.median(numeric_only=True))

# Mode
print("\nMode")
print(df.mode())

# Standard Deviation
print("\nStandard Deviation")
print(df.std(numeric_only=True))

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Iris Dataset Histogram")
plt.savefig("histogram.png")
plt.show()

# Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=df.iloc[:,0:4])
plt.title("Boxplot")
plt.savefig("boxplot.png")
plt.show()

# Correlation Heatmap
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()
print("EDA Completed Successfully")

