import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


df = pd.read_csv("C:\\Users\\junai\\OneDrive\\Desktop\\Dataset\\bengaluru_house_prices.csv")

df.drop(columns="society", inplace=True)

df1 = df.drop(['area_type', 'availability', 'balcony'], axis=1)

print("Missing values before cleaning:")
print(df1.isnull().sum())

def simple_avg(x):
    x = str(x)
    if '-' in x:
        parts = x.split('-')
        if len(parts) == 2:
            a, b = parts
            if a.replace('.', '', 1).isdigit() and b.replace('.', '', 1).isdigit():
                return (float(a) + float(b)) / 2
            else:
                return None
        else:
            return None
    elif x.replace('.', '', 1).isdigit():
        return float(x)
    else:
        return None

df1['total_sqft'] = df1['total_sqft'].apply(simple_avg)

df1 = df1.dropna(subset=['total_sqft'])


print(df1.isnull().sum())

df1.to_csv("cleaned_file.csv", index=False)
print(df1.head())

df2=df1.copy()
df2['price_per_sqft']=df2['price']*1000000/df2['total_sqft']
df2=df2.head()
print(df2)

df2_stats=df2['price_per_sqft'].describe()
print(df2_stats)

q1 = df2['price_per_sqft'].quantile(0.25)
q3 = df2['price_per_sqft'].quantile(0.75)
iqr = q3 - q1

# Filter out outliers
df2 = df2[(df2['price_per_sqft'] >= q1 - 1.5 * iqr) & (df2['price_per_sqft'] <= q3 + 1.5 * iqr)]
print(df2.head())
print("\nData after removing outliers (IQR method):")
print(df2.head(10))  

# Distribution Plot of Price (in millions)

# sns.histplot(df2['price'], bins=30, kde=True)
# plt.title("Distribution of House Prices")
# plt.xlabel("Price (Millions)")
# plt.ylabel("Frequency")
# plt.show()

# Count Plot of Number of BHKs

# sns.countplot(x='size', data=df2)
# plt.xticks(rotation=45)
# plt.title("Number of Bedrooms (BHK)")
# plt.show()

# Boxplot of Price per Sqft by Location (Top 10)
top_locations = df2['location'].value_counts().nlargest(10).index
sns.boxplot(x='location', y='price_per_sqft', data=df2[df2['location'].isin(top_locations)])
plt.xticks(rotation=45)
plt.title("Price/Sqft by Top 10 Locations")
plt.show()

plt.scatter(df2['total_sqft'], df2['price'], alpha=0.5)
plt.xlabel("Total Sqft")
plt.ylabel("Price")
plt.title("Total Sqft vs Price")
plt.show()

# . Bar Chart of Average Price by Location
avg_price_location = df2.groupby('location')['price'].mean().sort_values(ascending=False)[:10]
avg_price_location.plot(kind='bar', figsize=(10,5), title='Average Price by Location')
plt.ylabel("Average Price (Millions)")
plt.show()
