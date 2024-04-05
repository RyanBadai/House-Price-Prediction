import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv("housePrice.csv")
df_org = df.copy()
df.head()

df.shape

df.info()

df.describe()

# Mencari Missing Value
df.isna().sum()

df.dropna(subset=["Address"], inplace=True)

df.isna().sum()

df.shape

df['Parking'].value_counts(normalize = True)*100

pd.crosstab(df['Parking'],df['Room'])

df.groupby(['Parking','Room'])['Price'].mean()

df.groupby(['Elevator','Warehouse'])['Price'].mean()

df.head()

df["Area"] = pd.to_numeric(df["Area"] , errors='coerce')

nulls_Area_index = np.where(df["Area"].isna())[0]
print(f"The index of nulls in Area is :{nulls_Area_index}")
for index in nulls_Area_index:
    print(df_org.Area[index])
    
df['Area'] = df_org['Area'].apply(lambda X : re.sub(',','',X))
df["Area"] = pd.to_numeric(df["Area"] , errors='coerce')
df.Area.dtype

df.dtypes

df.drop(columns = ['Price(USD)'], inplace=True)
# Karena kolom Price dan Price(USD) mempunyai nilai yang sama, maka kolom Price(USD) saya buang
Boolean_ft = ['Parking','Warehouse','Elevator']

df[Boolean_ft] = df[Boolean_ft].astype('int64')

df.sample(5)

print("skew feature: \n",df.drop('Address',axis='columns',inplace=False).skew())

sns.boxplot(x=df['Area'])

sns.boxplot(x=df['Price'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
df_clean = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah saya drop outliers
df_clean.shape

# Melihat kolerasi antar variabel
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

sns.barplot(df_clean, x=df.Room, y=df.Price)
plt.show()

#One Hot Encoding
address_dummy = pd.get_dummies(df_clean['Address'])
df_clean = df.merge(address_dummy, left_index = True, right_index = True)
df_clean.drop(columns = 'Address', inplace = True)
df_clean.head(3)

from sklearn.model_selection import train_test_split
 
X = df_clean.drop(["Price"],axis =1)
y = df_clean["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state=123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# Normalisasi
from sklearn.preprocessing import StandardScaler

numerical_features = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
acc = pd.DataFrame(index=['accuracy'])

knn = KNeighborsRegressor(n_neighbors = 1)
knn.fit(X_train, y_train)
acc.loc['accuracy', 'knn'] = knn.score(X_test,y_test)
knn.score(X_test,y_test)

rf = RandomForestRegressor(n_estimators = 50, max_depth = 16, random_state = 11)
rf.fit(X_train, y_train)
acc.loc['accuracy', 'rf'] = rf.score(X_test,y_test)
rf.score(X_test,y_test)

boosting = AdaBoostRegressor(n_estimators = 25, learning_rate = 0.001, random_state = 11)                             
boosting.fit(X_train, y_train)
acc.loc['accuracy', 'boosting'] = boosting.score(X_test,y_test)
boosting.score(X_test,y_test)

acc

# Menghitung Nilai MSE
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','Random Forest','Boosting'])
model_dict = {'KNN': knn, 'Random Forest':rf, 'Boosting': boosting}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[5:10].copy()
pred_dict = {'y_true':y_test[5:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)