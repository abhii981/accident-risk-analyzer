import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('data/indian_roads_dataset.csv')
df['festival']=df['festival'].fillna('none')
df['date']=pd.to_datetime(df['date'])
df['month']=df['date'].dt.month
cat_cols=['city','state','weather','road_type','visibility','traffic_density','festival','cause','day_of_week']
le=LabelEncoder()
for col in cat_cols:
    df[col + '_enc']=le.fit_transform(df[col])

severity_map={'minor':0, 'major':1, 'fatal':2}
df['severity_label']=df['accident_severity'].map(severity_map)

#print(df.shape)
print(df.isnull().sum().sum(), "missing values remaining")
#print(df.head(3))
#for col in df.columns