import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

DATA_FILE = "home_data.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print("Data loaded from file.")
    else:
        # Initial data
        df = pd.DataFrame({
            "area": [100, 150, 200, 120],
            "rooms": [5, 10, 20, 8],
            "price": [500, 700, 1000, 550]
        })
        df.to_csv(DATA_FILE, index=False)
        print("New data file created.")
    return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)
    print("Data saved successfully.")

def add_new_data(df):
    print("Add new data (leave area empty to exit)")
    while True:
        area = input("Area (sqm): ")
        if area.strip() == "":
            break
        rooms = input("Number of rooms: ")
        price = input("Price (million): ")
        try:
            df = pd.concat([df, pd.DataFrame([{
                "area": float(area),
                "rooms": int(rooms),
                "price": float(price)
            }])], ignore_index=True)
        except Exception as e:
            print(f"Invalid input! Try again. ({e})")
    return df

def plot_data(df):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(df["area"], df["price"], c='blue', edgecolors='k')
    plt.xlabel("Area (sqm)")
    plt.ylabel("Price (million)")
    plt.title("Area vs Price")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.subplot(1,2,2)
    plt.scatter(df["rooms"], df["price"], color='orange', edgecolors='k')
    plt.xlabel("Rooms")
    plt.ylabel("Price (million)")
    plt.title("Rooms vs Price")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def show_model_coefficients(model, feature_names):
    print("\nModel coefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

def predict_prices(model):
    try:
        n = int(input("How many new houses to predict? "))
        for i in range(n):
            area = float(input(f"Area of house {i+1} (sqm): "))
            rooms = int(input(f"Number of rooms in house {i+1}: "))
            price = model.predict([[area, rooms]])
            print(f"Predicted price for house {i+1}: {price[0]:.2f} million")
    except Exception as e:
        print(f"Invalid input! ({e})")

def clean_data(df):
    before = len(df)
    # Convert columns to numeric, coerce errors to NaN
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df_clean = df.dropna(subset=["area", "rooms", "price"])
    after = len(df_clean)
    if after < before:
        print(f"Warning: {before - after} incomplete or invalid row(s) were removed.")
    return df_clean

def main():
    print("=== Home Price Analysis System ===")
    df = load_data()
    while True:
        print("\n1. Add new data\n2. Show plots\n3. Predict price\n4. Show model coefficients\n5. Exit")
        choice = input("Select an option: ")
        if choice == "1":
            df = add_new_data(df)
            save_data(df)
        elif choice == "2":
            plot_data(df)
        elif choice == "3":
            df_clean = clean_data(df)
            if len(df_clean) < 2:
                print("Not enough valid data to train the model.")
                continue
            try:
                model = LinearRegression()
                X = df_clean[["area", "rooms"]].values
                y = df_clean["price"].values
                model.fit(X, y)
                predict_prices(model)
            except Exception as e:
                print(f"Error during prediction: {e}")
        elif choice == "4":
            df_clean = clean_data(df)
            if len(df_clean) < 2:
                print("Not enough valid data to train the model.")
                continue
            try:
                model = LinearRegression()
                X = df_clean[["area", "rooms"]].values
                y = df_clean["price"].values
                model.fit(X, y)
                show_model_coefficients(model, ["Area", "Rooms"])
            except Exception as e:
                print(f"Error during model training: {e}")
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid selection! Please try again.")

if __name__ == "__main__":
    main()
