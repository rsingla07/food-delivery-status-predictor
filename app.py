import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load the saved model and preprocessing objects
# (these 6 files must be in the same folder as this app.py)
# ------------------------------------------------------------
model = joblib.load("delivery_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
priority_map = joblib.load("priority_map.pkl")
features_to_keep = joblib.load("features_to_keep.pkl")
median_time = joblib.load("median_time.pkl")

st.title("🍔 Food Delivery Status Predictor")
st.write("Enter order details to predict whether the delivery will be **Fast** or **Delayed**.")

# ------------------------------------------------------------
# Input widgets - one per feature the model needs
# ------------------------------------------------------------
distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=10.0)
weather = st.selectbox("Weather Conditions", encoders['Weather_Conditions'].classes_)
traffic = st.selectbox("Traffic Conditions", encoders['Traffic_Conditions'].classes_)
experience = st.number_input("Delivery Person Experience (years)", min_value=0, max_value=20, value=5)
priority = st.selectbox("Order Priority", list(priority_map.keys()))
order_time = st.selectbox("Order Time", encoders['Order_Time'].classes_)
vehicle = st.selectbox("Vehicle Type", encoders['Vehicle_Type'].classes_)
restaurant_rating = st.slider("Restaurant Rating", 1.0, 5.0, 4.0)
customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0)
order_cost = st.number_input("Order Cost", min_value=0.0, max_value=5000.0, value=500.0)
tip_amount = st.number_input("Tip Amount", min_value=0.0, max_value=500.0, value=30.0)

if st.button("Predict Delivery Status"):
    # Build a single-row DataFrame matching the training feature order
    input_dict = {
        'Distance': distance,
        'Weather_Conditions': encoders['Weather_Conditions'].transform([weather])[0],
        'Traffic_Conditions': encoders['Traffic_Conditions'].transform([traffic])[0],
        'Delivery_Person_Experience': experience,
        'Order_Priority': priority_map[priority],
        'Order_Time': encoders['Order_Time'].transform([order_time])[0],
        'Vehicle_Type': encoders['Vehicle_Type'].transform([vehicle])[0],
        'Restaurant_Rating': restaurant_rating,
        'Customer_Rating': customer_rating,
        'Order_Cost': order_cost,
        'Tip_Amount': tip_amount,
    }
    input_df = pd.DataFrame([input_dict])

    # Scale the same numeric columns that were scaled during training
    num_cols = ['Order_Cost', 'Distance', 'Delivery_Person_Experience',
                'Restaurant_Rating', 'Customer_Rating', 'Tip_Amount']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Reorder columns to match training exactly
    input_df = input_df[features_to_keep]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"Predicted: **Delayed** (probability: {probability:.2%})")
    else:
        st.success(f"Predicted: **Fast** (probability of delay: {probability:.2%})")

    st.caption(f"Reference median delivery time in training data: {median_time:.1f} minutes")
