import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

df = pd.read_csv('train.csv')

cat_cols = ['Destination', 'HasFamily', 'VIP', 'HomePlanet', 'Side', 'Deck', 'CryoSleep']
num_cols = ['VRDeck', 'ShoppingMall', 'GroupSize', 'Spa', 'FoodCourt', 'RoomService', 'Age', 'CabinID', 'TotalExpense']

def preprocess(df):
    if 'Transported' in df.columns:
        df = df.drop(columns=['Transported'])
        
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['HasFamily'] = np.where(df['GroupSize'] > 1, True, False)
    df = df.drop(columns=['Name', 'PassengerId', 'Group'])

    df['Side'] = df['Cabin'].str[-1:]
    df['Deck'] = df['Cabin'].str[:1]
    df['CabinID'] = df['Cabin'].str.extract(r'(\d+)')
    df['CabinID'] = pd.to_numeric(df['CabinID'])
    df = df.drop(columns=['Cabin'])

    amenities = ['VRDeck', 'ShoppingMall', 'Spa', 'FoodCourt', 'RoomService']
    df.loc[df['CryoSleep'] == True, amenities] = df.loc[df['CryoSleep'] == True, amenities].fillna(0)
    
    df['TotalExpense'] = df[amenities].sum(axis=1)
    
    return df

X = preprocess(df)
Y = df['Transported'].astype(int) 

cat_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
)

num_transformer = make_pipeline(
    KNNImputer(n_neighbors=5)
)

preprocessing = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_cols),
        ('num', num_transformer, num_cols)
    ]
)

model = make_pipeline(
    preprocessing,
    XGBClassifier(random_state=3335, learning_rate=0.01, max_depth=7, n_estimators=400, n_jobs=-1)
)

model.fit(X, Y)

test_df = pd.read_csv('test.csv')
passengerIds = test_df['PassengerId']
test_X = preprocess(test_df)

results = model.predict(test_X)
results_df = pd.DataFrame({'PassengerId': passengerIds, 'Transported': results.astype(bool)})
results_df.to_csv('submission.csv', index=False)
print("Submission saved successfully!")