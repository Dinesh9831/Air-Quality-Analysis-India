
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

sns.set_style("whitegrid")


df = pd.read_csv(r"C:\Users\LENOVO\Desktop\BT\ProjectD\Datasets\Air Quality index india.csv")

print("\n Data Loaded Successfully")




print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- LAST 5 ROWS ---")
print(df.tail())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe(include='all'))




numerical_cols = ['latitude', 'longitude', 'pollutant_min', 'pollutant_max', 'pollutant_avg']
categorical_cols = ['pollutant_id']

df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("\n Data Cleaned | Shape:", df.shape)






preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
])

X_processed = preprocessor.fit_transform(df)

ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(ohe_cols)

X = pd.DataFrame(X_processed, columns=feature_names)
y = df['pollutant_avg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)





pollutant_share=df.groupby('pollutant_id')['pollutant_avg'].sum()
donut_colors=plt.cm.Set3(np.linspace(0,1,len(pollutant_share)))
plt.figure(figsize=(8,8))
plt.pie(pollutant_share.values,labels=pollutant_share.index,autopct='%1.1f%%',startangle=140,colors=donut_colors,wedgeprops=dict(width=0.4))
plt.title("Pollutant-wise Contribution to Overall Pollution")
plt.show()



top_cities_combo=df.groupby('city')['pollutant_avg'].mean().nlargest(8)
fig,ax1=plt.subplots(figsize=(12,6))
bar_colors=plt.cm.Pastel1(np.linspace(0,1,len(top_cities_combo)))
ax1.bar(top_cities_combo.index,top_cities_combo.values,color=bar_colors,label="Average Pollution")
ax1.set_xlabel("City")
ax1.set_ylabel("Average Pollution Level")
ax1.tick_params(axis='x',rotation=45)
ax2=ax1.twinx()
ax2.plot(top_cities_combo.index,top_cities_combo.values,color="#2c3e50",marker='o',linewidth=2,label="Trend")
ax2.set_ylabel("Pollution Trend")
plt.title("Top Cities: Average Pollution (Bar + Trend Line)")
fig.tight_layout()
plt.show()




plt.figure(figsize=(10,8))
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f",cmap="magma")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numerical_cols])
plt.title("Box Plot of Numerical Features")
plt.show()

top_cities = df.groupby('city')['pollutant_avg'].mean().nlargest(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_cities.index, y=top_cities.values ,palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Cities by Average Pollution")
plt.show()

top_states = df.groupby('state')['pollutant_avg'].mean().nlargest(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_states.index, y=top_states.values,  palette="cubehelix")
plt.xticks(rotation=45)
plt.title("Top 10 States by Average Pollution")
plt.show()


poll_type_avg = df.groupby('pollutant_id')['pollutant_avg'].mean().sort_values()
colors = plt.cm.viridis(np.linspace(0, 1, len(poll_type_avg)))
plt.figure(figsize=(10,6))
plt.barh(poll_type_avg.index,poll_type_avg.values,color=colors)
plt.xlabel("Average Pollutant Value")
plt.ylabel("Pollutant Type")
plt.title("Average Pollutant by Type (Horizontal Bar Chart)")
plt.show()
 

plt.figure(figsize=(12,8))
plt.scatter(df['longitude'], df['latitude'],c=df['pollutant_avg'],s=df['pollutant_avg'] / 3,alpha=0.6)
plt.colorbar(label="Pollutant Average")
plt.title("Geographical Pollution Heatmap")
plt.show()



lr = LinearRegression()
lr.fit(X_train, y_train)
print("\nLinear Regression R2:", r2_score(y_test, lr.predict(X_test)))

lasso = Lasso(max_iter=5000)
lasso.fit(X_train, y_train)
print("Lasso Regression R2:", r2_score(y_test, lasso.predict(X_test)))

dt = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=50,
    random_state=42
)
dt.fit(X_train, y_train)
print("Decision Tree R2:", r2_score(y_test, dt.predict(X_test)))

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
print("Random Forest R2:", r2_score(y_test, rf.predict(X_test)))

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
print("Gradient Boosting R2:", r2_score(y_test, gb.predict(X_test)))


plt.figure(figsize=(28, 14))
plot_tree(dt,feature_names=feature_names,filled=True,rounded=True,fontsize=9,impurity=False,precision=2)
plt.title("Decision Tree Structure (Depth = 4)", fontsize=16)
plt.show()





X_unsup = df[numerical_cols]

kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans'] = kmeans.fit_predict(X_unsup)

pca = PCA(n_components=3)
pca.fit_transform(X_unsup)
print("\nPCA Explained Variance:", pca.explained_variance_ratio_.sum())

agglo = AgglomerativeClustering(n_clusters=5)
df['Agglo'] = agglo.fit_predict(X_unsup)

dbscan = DBSCAN(eps=0.5)
df['DBSCAN'] = dbscan.fit_predict(X_unsup)

iso = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = iso.fit_predict(X_unsup)





output_cols = [
    'country', 'state', 'city', 'station',
    'pollutant_id', 'pollutant_avg',
    'KMeans', 'Agglo', 'DBSCAN', 'Anomaly'
]

df[output_cols].to_csv("airquality_results_final.csv", index=False)

print("\n Final Output Saved: airquality_results_final.csv")
