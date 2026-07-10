import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Show files in current folder
print("Files in folder:")
print(os.listdir())

# Load Dataset
df = pd.read_csv("iris.csv")

# -------------------------
# Bar Plot
# -------------------------
plt.figure(figsize=(8,6))

sns.countplot(
    x='species',
    data=df
)

plt.title("Count of Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")

plt.savefig("barplot.png")
plt.show()

# -------------------------
# Line Chart
# -------------------------
plt.figure(figsize=(10,6))

plt.plot(
    df.index,
    df['sepal_length']
)

plt.title("Sepal Length Trend")
plt.xlabel("Record Number")
plt.ylabel("Sepal Length")

plt.savefig("linechart.png")
plt.show()

# -------------------------
# Scatter Plot
# -------------------------
plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x='sepal_length',
    y='petal_length',
    hue='species'
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.savefig("scatterplot.png")
plt.show()

print("Task 3 Completed Successfully!")