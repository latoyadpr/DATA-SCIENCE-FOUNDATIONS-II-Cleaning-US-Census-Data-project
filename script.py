import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecademylib3_seaborn

import glob

files = glob.glob("states*.csv")

df_list = []
for filename in files:
  data = pd.read_csv(filename)
  df_list.append(data)

df = pd.concat(df_list)

print(df.columns)
print(df.dtypes)


print(df.head())

df['Income'] = df['Income'].str.replace(r'\$', '', regex=True)
df['Income'] = pd.to_numeric(df['Income'])


df[['Men', 'Women']] = df['GenderPop'].str.split('_', expand=True)


df['Men'] = df['Men'].str.replace('M', '')
df['Women'] = df['Women'].str.replace('F', '')

df['Men'] = pd.to_numeric(df['Men'])
df['Women'] = pd.to_numeric(df['Women'])


plt.scatter(df['Women'], df['Income'])
plt.xlabel('Number of Women')
plt.ylabel('Average Income')
plt.title('Average Income vs. Proportion of Women in Each State')
plt.show()

print(df['Women'])

df['Women'] = df['Women'].fillna(df['TotalPop'] - df['Men'])

print(df['Women'])

duplicates = df.duplicated()
print(duplicates)


df = df.drop_duplicates()

plt.scatter(df['Women'], df['Income'])
plt.xlabel('Number of Women')
plt.ylabel('Average Income')
plt.title('Average Income vs. Proportion of Women in Each State')
plt.show()



plt.scatter(df['Women'], df['Income'])
plt.xlabel('Number of Women')
plt.ylabel('Average Income')
plt.title('Average Income vs. Proportion of Women in Each State')
plt.show()

print(df.columns)


# Convert race columns to numerical types
race_columns = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']
for column in race_columns:
    df[column] = df[column].str.replace('%', '').astype(float)
    df[column] = df[column].fillna(df[column].mean())  # Fill NaNs with the mean of the column

# Create histograms
for column in race_columns:
    plt.hist(df[column], bins=10, edgecolor='black')
    plt.title(f'Distribution of {column} Population')
    plt.xlabel(f'{column} Percentage')
    plt.ylabel('Number of States')
    plt.show()

import seaborn as sns

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.boxplot(x='State', y='Income', data=df)
plt.xticks(rotation=90)
plt.title('Income Distribution by State')
plt.show()

