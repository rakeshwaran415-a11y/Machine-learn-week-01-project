import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

# Independent variable (X) and Dependent variable (y)
X = df[['AI_Exposure_Index']]          # Feature
y = df['Automation_Probability_2030']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get equation values
m = model.coef_[0]
c = model.intercept_

print(f"Linear Regression Equation:")
print(f"Automation Probability = {m:.2f} × AI Exposure Index + {c:.2f}")

# User input for prediction
user_ai_exposure = float(input("Enter AI Exposure Index (0 to 1): "))
user_input_df = pd.DataFrame({'AI_Exposure_Index': [user_ai_exposure]})

# Prediction
predicted_probability = model.predict(user_input_df)
print(f"Predicted Automation Probability in 2030: {predicted_probability[0]:.2f}")
