import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('female-to-male-life-expectancy-ratio.csv')
print(df.head())
df.dropna(inplace=True) 

print(df.info())


X = df.drop(columns=['Entity', 'Code', 'Life expectancy ratio (f/m) - Type: period - Sex: both - Age: 0'])
y = df['Life expectancy ratio (f/m) - Type: period - Sex: both - Age: 0']  # this is the target column
#Spliting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

print("Model training complete.")
#Prediction 

label_encoder = LabelEncoder()
df['Entity'] = label_encoder.fit_transform(df['Entity'])

new_data = pd.DataFrame({
    'Year': [2021],
    'Entity': label_encoder.transform(['Afghanistan']),
    'Life expectancy ratio (f/m) - Type: period - Sex: both - Age: 15': [1.15],
    'Life expectancy ratio (f/m) - Type: period - Sex: both - Age: 45': [1.03], 
})
new_data = new_data[['Year', 
                     'Life expectancy ratio (f/m) - Type: period - Sex: both - Age: 15', 
                     'Life expectancy ratio (f/m) - Type: period - Sex: both - Age: 45']] 


prediction = model.predict(new_data)

# Output the result
print("Predicted Life Expectancy Ratio:", prediction)

# Evaluate the model
test_score = model.score(X_test, y_test)
print(f"Model Test Score (R^2): {test_score}")

# Calculate Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")