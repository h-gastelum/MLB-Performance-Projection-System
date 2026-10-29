
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

# Load the CSV file
df = pd.read_csv("aaron_judge_stats.csv")

# Sort seasons from oldest to newest
df = df.sort_values("season").reset_index(drop=True)

# Make sure the important columns are numbers
columns = ["season", "age", "gamesPlayed", "homeRuns",
           "hits", "rbi", "strikeOuts", "atBats"]

for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Use this season's stats to predict next season's home runs
df["next_year_hr"] = df["homeRuns"].shift(-1)

# Create input features
features = [
    "age",
    "gamesPlayed",
    "homeRuns",
    "hits",
    "rbi",
    "strikeOuts",
    "atBats"
]

# Remove rows with missing values
df = df.dropna(subset=features + ["next_year_hr"])

# Use earlier seasons for training and later seasons for testing
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

if len(train) < 2 or len(test) < 1:
    print("Not enough seasons to train and test the model.")
else:
    X_train = train[features]
    y_train = train["next_year_hr"]

    X_test = test[features]
    y_test = test["next_year_hr"]

    # Create and train the model
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict next-season home runs
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)

    print("Actual vs Predicted Home Runs")

    for actual, predicted in zip(y_test, predictions):
        print(f"Actual: {actual:.0f}, Predicted: {predicted:.1f}")

    print(f"\nMean Absolute Error: {mae:.2f}")

    # Train on all available examples and save the model
    model.fit(df[features], df["next_year_hr"])
    #model.save_model("mlb_home_run_model.json")

    #print("Model saved to mlb_home_run_model.json")
