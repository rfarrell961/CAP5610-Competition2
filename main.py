import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

train.drop(["PassengerId", "Name"], axis=1, inplace=True)
test.drop(["Name"], axis=1, inplace=True)


# Handle Missing values and convert to int for CryoSleep
train['CryoSleep'].fillna(False, inplace=True)
train["CryoSleep"] = train["CryoSleep"].astype(int)
test['CryoSleep'].fillna(False, inplace=True)
test["CryoSleep"] = test["CryoSleep"].astype(int)

# Handle Missing values and convert to int for VIP
train['VIP'].fillna(False, inplace=True)
train["VIP"] = train["VIP"].astype(int)
test['VIP'].fillna(False, inplace=True)
test["VIP"] = test["VIP"].astype(int)

# Handle HomePlanet Missing Values
train['HomePlanet'].fillna(train['HomePlanet'].mode().iloc[0], inplace=True)
train['Destination'].fillna(train['Destination'].mode().iloc[0], inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['HomePlanet'].fillna(test['HomePlanet'].mode().iloc[0], inplace=True)
test['Destination'].fillna(test['Destination'].mode().iloc[0], inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)


# Handle RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck missing values
combine = [train, test]
for df in combine:
    df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)
    df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=True)
    df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)
    df['Spa'].fillna(df['Spa'].mean(), inplace=True)
    df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)


# Parse Cabin Info, and handle missing values
combine = [train, test]
for df in combine:
    df["deck"] = (df["Cabin"].str.split("/", n=3, expand=False)).str[0]
    df["side"] = (df["Cabin"].str.split("/", n=3, expand=False)).str[2]
    df['deck'].fillna(df['deck'].mode().iloc[0], inplace=True)
    df['side'].fillna(df['side'].mode().iloc[0], inplace=True)
    df.drop("Cabin", axis=1, inplace=True)

train = pd.get_dummies(train, columns=["HomePlanet", "Destination", "deck", "side"], prefix=["Home", "Dest", "deck", "side"])
test = pd.get_dummies(test, columns=["HomePlanet", "Destination", "deck", "side"], prefix=["Home", "Dest", "deck", "side"])

train_x = train.drop("Transported", axis=1)
train_y = train["Transported"]
test_x = test.drop("PassengerId", axis=1).copy()

"""
# Hyper Parameter Tuning
max_depth = [x for x in range(10, 100, 10)]
min_sample_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
n_estimators = [x for x in range(100, 1000, 100)]
bootstrap = [True, False]
criterion = ['gini', 'entropy', 'log_loss']

random_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'max_depth': max_depth,
    'min_samples_split': min_samples_leaf,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=300, cv=5, verbose=2, random_state=0, n_jobs=-1)
rf_random.fit(train_x, train_y)
print(rf_random.best_params_)
"""

# Random Forest
random_forest = RandomForestClassifier(n_estimators=700, min_samples_split=4, min_samples_leaf=4, max_depth=30, criterion='log_loss', bootstrap=True)
random_forest.fit(train_x, train_y)
Y_pred = random_forest.predict(test_x)
random_forest.score(train_x, train_y)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)
print(f"Training Accuracy {acc_random_forest}")

submission = pd.DataFrame({
    "PassengerID": test["PassengerId"],
    "Transported": Y_pred
})

submission.to_csv('submission.csv', index=False)


