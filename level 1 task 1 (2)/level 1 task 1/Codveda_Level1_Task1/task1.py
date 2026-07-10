import pandas as pd

# Load dataset
df = pd.read_csv("3) Sentiment dataset.csv")

# Show first rows
print(df.head())

# Dataset info
print(df.info())

# Missing values
print(df.isnull().sum())

# Duplicate rows
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Save cleaned dataset
df.to_csv("cleaned_sentiment_dataset.csv", index=False)

print("Cleaning completed successfully!")
