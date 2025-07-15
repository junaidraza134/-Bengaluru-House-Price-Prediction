# Bengaluru House Price Prediction
# 🏡 Bengaluru House Price Prediction

## 📌 Problem Statement

Build a data science solution to predict housing prices in Bengaluru using variables such as:
📍 Property location  
📐 Built-up area  
🛏️ Number of bedrooms  
📅 Property age  
🚉 Access to public amenities

Use **regression techniques** to estimate property value and analyze pricing drivers. Present findings through **Power BI**, highlighting neighborhood price trends and key value influencers for 🏘️ buyers and 🏗️ developers.

---

## 🎯 Project Objective

- 📊 Predict house prices in Bengaluru based on structured property data  
- 🧹 Preprocess data to prepare clean and meaningful inputs  
- 🧠 Analyze important features that influence pricing  
- 📉 Visualize trends and insights using Power BI dashboards  
- 🧭 Support decision-making for buyers, sellers, and real estate planners  

---

## 🔄 Workflow

1. 📥 **Data Collection**  
   Dataset includes features like location, area (sqft), BHK, bathrooms, etc.

2. 🧼 **Data Preprocessing**
   - ❌ Handle missing values
   - 🔁 Remove duplicates
   - 🚫 Detect and remove outliers
   - 🔡 Encode categorical features
   - 📏 Scale numerical data

3. 🕵️‍♂️ **Exploratory Data Analysis (EDA)**
   - 📊 Visualize distributions, trends, and outliers
   - 🔍 Analyze feature correlation
   - 🗺️ Study neighborhood-wise price distribution

4. 🧮 **Model Building**
   - ⚙️ Feature Engineering
   - 🤖 Applied Models:
     - 📈 Linear Regression
     - 🧵 Lasso / Ridge Regression
     - 🌲 Decision Tree Regressor
     - 🌳 Random Forest Regressor
   - 📐 Evaluation Metrics: R² Score, MAE, MSE, RMSE

5. 📊 **Insights & Visualization**
   - 🧩 Create an interactive dashboard in **Power BI**
   - 🗺️ Show location-wise price variations and key influencers

---

## 🛠️ Tools & Technologies Used

- 🐍 **Python** – NumPy, Pandas, Seaborn, Matplotlib, Scikit-Learn  
- 📓 **Jupyter Notebook** – Code development and analysis  
- 📊 **Power BI** – Dashboard and storytelling  
- 🌐 **GitHub** – Version control and project collaboration  

---

## 📈 Key Insights

- 📍 Locations like **Whitefield**, **Indiranagar**, and **Marathahalli** show higher average prices.
- 🛁 **Number of bathrooms** and **total area** are stronger predictors than BHK count.
- ⚠️ Several outliers with unrealistically high sqft were removed to improve model accuracy.
- 📉 Ridge Regression offered better generalization with reduced overfitting compared to plain Linear Regression.

---
