import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Height & Weight Predictor", page_icon="📏")

st.title("📏 Age se Height & Weight Predictor")
st.write("Simple Linear Regression | Dataset: Kids198 (Age 8-18 years)")

# Dataset load
df = pd.read_csv("weight-height.csv")

# Age months → years
df['Age_years'] = df['Age'] / 12                     # months ko year me convert kiya

# Height inches → cm
df['Height_cm'] = df['Height'] * 2.54                # 1 inch = 2.54 cm

# Weight pounds → kg
df['Weight_kg'] = df['Weight'] * 0.453592            # 1 pound = 0.453592 kg

# Features
X = df[['Age_years']]                                # Input feature (Age)

# Targets
y_height = df['Height_cm']                           # Target 1 - Height
y_weight = df['Weight_kg']                           # Target 2 - Weight

# Split (Height ke liye)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X, y_height, test_size=0.2, random_state=42
)

# Split (Weight ke liye)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X, y_weight, test_size=0.2, random_state=42
)

# -------------------- Height Model --------------------
model_height = LinearRegression()                    # Height model banaya
model_height.fit(X_train_h, y_train_h)               # Height model train kiya

# -------------------- Weight Model --------------------
model_weight = LinearRegression()                    # Weight model banaya
model_weight.fit(X_train_w, y_train_w)               # Weight model train kiya

# Model scores
r2_height = r2_score(y_test_h, model_height.predict(X_test_h))
r2_weight = r2_score(y_test_w, model_weight.predict(X_test_w))

# Sidebar
st.sidebar.header("Model Performance")
st.sidebar.write(f"Height R² Score: **{r2_height:.3f}**")
st.sidebar.write(f"Weight R² Score: **{r2_weight:.3f}**")
st.sidebar.write("Valid Age Range: **8 - 18 years**")

# User Input
st.subheader("Enter Age to Predict")
age = st.number_input("Age (in years)", min_value=5.0, max_value=25.0, value=12.0, step=0.5)

if st.button("Predict"):
    if age < 8 or age > 18:
        st.warning("⚠️ Yeh model sirf 8 se 18 saal ke liye train hua hai. Prediction galat ho sakti hai.")
    
    pred_height = model_height.predict([[age]])[0]   # Height predict kiya
    pred_weight = model_weight.predict([[age]])[0]   # Weight predict kiya

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Predicted Height**")
        st.write(f"### {pred_height:.1f} cm")
        st.caption(f"≈ {pred_height/2.54:.1f} inches")

    with col2:
        st.success(f"**Predicted Weight**")
        st.write(f"### {pred_weight:.1f} kg")
        st.caption(f"≈ {pred_weight/0.453592:.1f} pounds")

# Graph
st.subheader("Model Graph")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Height graph
axes[0].scatter(X_test_h, y_test_h, color='blue', alpha=0.6, label='Actual')
axes[0].plot(X_test_h, model_height.predict(X_test_h), color='red', label='Predicted')
axes[0].set_xlabel("Age (Years)")
axes[0].set_ylabel("Height (cm)")
axes[0].set_title("Age vs Height")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Weight graph
axes[1].scatter(X_test_w, y_test_w, color='green', alpha=0.6, label='Actual')
axes[1].plot(X_test_w, model_weight.predict(X_test_w), color='orange', label='Predicted')
axes[1].set_xlabel("Age (Years)")
axes[1].set_ylabel("Weight (kg)")
axes[1].set_title("Age vs Weight")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.caption("Built with Linear Regression | Dataset: Kids198")
