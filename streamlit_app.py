import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title=" Smart House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 1.5rem;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 2rem;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    h1 {
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    model_path = 'house_price_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Make sure 'house_price_model.pkl' is in the same folder as this script.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_locations():
    """Extract available locations from the training data"""
    csv_path = 'bengaluru_house.csv'
    if not os.path.exists(csv_path):
        st.error(f"❌ Data file not found: {csv_path}")
        st.stop()
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['location'])
    df['location'] = df['location'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    location_stats = df['location'].value_counts()
    location_less_than_10 = location_stats[location_stats <= 10].index.tolist()
    
    unique_locations = set()
    for loc in df['location']:
        if loc in location_less_than_10:
            unique_locations.add('other')
        else:
            unique_locations.add(loc)
    
    unique_locations = [loc for loc in unique_locations if isinstance(loc, str)]
    return sorted(list(unique_locations))

def get_area_types():
    """Get available area types"""
    return ['Residential', 'Commercial', 'Super built-up  Area']

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def create_feature_vector(total_sqft, bath, bhk, location, area_type, feature_columns):
    """Create feature vector matching training preprocessing"""
    x = np.zeros(len(feature_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    
    area_type_col = f'area_type_{area_type}'
    if area_type_col in feature_columns:
        idx = list(feature_columns).index(area_type_col)
        x[idx] = 1
    
    if location != 'other':
        location_col = f'{location}'
        if location_col in feature_columns:
            idx = list(feature_columns).index(location_col)
            x[idx] = 1
    
    return x

# ============================================================================
# INTERACTIVE SELECTORS
# ============================================================================
def get_bhk_emoji(bhk):
    """Return emoji representation for BHK"""
    emojis = {1: "🏠", 2: "🏡", 3: "🏢", 4: "🏬", 5: "🏭"}
    return emojis.get(bhk, "🏘️")

def create_bhk_selector():
    """Create interactive BHK selection with buttons"""
    st.subheader("🛏️ Select Number of Bedrooms (BHK)")
    cols = st.columns(5)
    
    for i in range(1, 6):
        with cols[i-1]:
            if st.button(f"{get_bhk_emoji(i)}\n{i} BHK", key=f"bhk_{i}", use_container_width=True):
                st.session_state.selected_bhk = i
    
    if 'selected_bhk' not in st.session_state:
        st.session_state.selected_bhk = 2
    
    return st.session_state.selected_bhk

def create_bathroom_selector():
    """Create interactive bathroom selection with buttons"""
    st.subheader("🚿 Select Number of Bathrooms")
    cols = st.columns(5)
    
    for i in range(1, 6):
        with cols[i-1]:
            if st.button(f"🚿 {i}", key=f"bath_{i}", use_container_width=True):
                st.session_state.selected_bath = i
    
    if 'selected_bath' not in st.session_state:
        st.session_state.selected_bath = 2
    
    return st.session_state.selected_bath

def create_area_type_selector():
    """Create interactive area type selection with buttons"""
    st.subheader("🏗️ Select Area Type")
    
    area_types = get_area_types()
    cols = st.columns(len(area_types))
    
    area_emojis = {
        'Residential': '🏘️',
        'Commercial': '🏢',
        'Super built-up  Area': '🌆'
    }
    
    for idx, area_type in enumerate(area_types):
        with cols[idx]:
            emoji = area_emojis.get(area_type, '🏗️')
            label = area_type.split()[0]
            if st.button(f"{emoji}\n{label}", key=f"area_{idx}", use_container_width=True):
                st.session_state.selected_area = area_type
    
    if 'selected_area' not in st.session_state:
        st.session_state.selected_area = 'Residential'
    
    return st.session_state.selected_area

def create_location_selector(locations):
    """Create interactive location selection with search"""
    st.subheader(" Select Location")
    
    search_term = st.text_input("🔍 Search Location:", placeholder="e.g., Whitefield, Bellandur...")
    filtered_locations = [loc for loc in locations if search_term.lower() in loc.lower()] if search_term else locations
    
    st.write(f"**Found {len(filtered_locations)} locations**")
    
    cols_per_row = 3
    for i in range(0, len(filtered_locations), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(filtered_locations):
                loc = filtered_locations[i + j]
                with col:
                    if st.button(f" {loc}", key=f"loc_{loc}", use_container_width=True):
                        st.session_state.selected_location = loc
    
    if 'selected_location' not in st.session_state:
        st.session_state.selected_location = locations[0]
    
    return st.session_state.selected_location

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 3rem; margin: 0;'>🏠 Smart House Price Predictor</h1>
        <p style='color: white; font-size: 1.1rem; margin-top: 0.5rem;'>Predict Bengaluru house prices using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    locations = load_locations()
    
    # Input Section with Tabs
    st.markdown("## Enter Property Details")
    
    tab1, tab2, tab3 = st.tabs(["🛏️ Property", " Location", "📐 Area"])
    
    with tab1:
        st.info("Select the number of bedrooms and bathrooms")
        bhk = create_bhk_selector()
        bath = create_bathroom_selector()
    
    with tab2:
        location = create_location_selector(locations)
        st.success(f" Selected: **{location}**")
    
    with tab3:
        area_type = create_area_type_selector()
        st.success(f" Selected: **{area_type}**")
    
    # Area Slider
    st.markdown("## 📏 Property Size (Total Sqft)")
    col1, col2 = st.columns([3, 1])
    with col1:
        total_sqft = st.slider(
            "Total Sqft",
            min_value=500,
            max_value=20000,
            value=1200,
            step=50,
            label_visibility="collapsed"
        )
    with col2:
        st.metric("Sqft", f"{total_sqft:,}")
    
    # Summary Cards
    st.markdown("## 📊 Property Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>🛏️</h3>
            <p style='font-size: 1.5rem; font-weight: bold;'>{bhk} BHK</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>🚿</h3>
            <p style='font-size: 1.5rem; font-weight: bold;'>{bath}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>📐</h3>
            <p style='font-size: 1.2rem; font-weight: bold;'>{total_sqft:,} Sqft</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>🏘️</h3>
            <p style='font-size: 0.9rem; font-weight: bold;'>{area_type.split()[0]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>📍</h3>
            <p style='font-size: 0.85rem; font-weight: bold;'>{location[:12]}{'...' if len(location) > 12 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction Button
    if st.button("🔮 Predict Price", use_container_width=True):
        try:
            # Validate input
            if total_sqft < 500:
                st.error("❌ Property size should be at least 500 Sqft for accurate predictions!")
                st.info("The model was trained on properties with a minimum of ~500 Sqft per BHK.")
                st.stop()
            
            n_features = model.n_features_in_
            feature_columns = ['total_sqft', 'bath', 'bhk', 'area_type_Commercial', 'area_type_Super built-up  Area']
            
            for loc in locations:
                if loc != 'other' and loc not in feature_columns:
                    feature_columns.append(loc)
            
            while len(feature_columns) < n_features:
                feature_columns.append(f'dummy_{len(feature_columns)}')
            feature_columns = feature_columns[:n_features]
            
            feature_vector = create_feature_vector(
                total_sqft, bath, bhk, location, area_type, feature_columns
            )
            prediction = model.predict([feature_vector])[0]
            
            # Display Prediction (Convert to Lakhs or Crores)
            price_lakhs = prediction * 10  # Model predicts in Crores, convert to Lakhs
            price_crores = prediction
            
            # Format display based on price - using only K (thousands) and L (lakhs)
            if price_lakhs >= 100:
                # Show in Lakhs
                display_price = f"₹ {price_lakhs:.1f}L"
            else:
                # Show in Thousands
                price_thousands = price_lakhs * 100
                display_price = f"₹ {price_thousands:.0f}K"
            
            st.markdown(f"""
            <div class='prediction-box'>
                💰 Predicted Price
                <br>
                <span style='font-size: 3rem;'>{display_price}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional Details
            with st.expander("📈 Detailed Breakdown"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    price_per_sqft = (price_crores * 10_000_000) / total_sqft
                    if price_per_sqft >= 100000:
                        st.metric("Price per Sqft", f"₹ {price_per_sqft/100000:.2f}L")
                    else:
                        st.metric("Price per Sqft", f"₹ {price_per_sqft:.0f}")
                with col2:
                    st.metric("Total in Lakhs", f"₹ {price_lakhs:.1f}L")
                with col3:
                    st.metric("Total in Thousands", f"₹ {price_lakhs * 100:.0f}K")
            
            st.info(
                f"✅ **Prediction successful!**\n\n"
                f"📍 Location: {location} | 🏗️ Area Type: {area_type}\n\n"
                f"🤖 Model: XGBoost | 📊 R² Score: 0.647 | 📉 RMSE: ₹92.95 Cr"
            )
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            st.warning("Please check that all inputs are valid.")
    
    # Footer Info
    st.divider()
    st.markdown("## ℹ️ How This Works")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🧠 Model Details
        - **Algorithm:** XGBoost Regressor
        - **Training R² Score:** 0.647
        - **RMSE:** ₹92.95 Crores
        - **Test Accuracy:** ~65%
        - **Features:** 150+ location variables + property attributes
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Input Features
        - **Bedrooms (BHK):** 1-5
        - **Bathrooms:** 1-5
        - **Total Area:** 300-20,000 Sqft
        - **Location:** 100+ Bengaluru areas
        - **Area Type:** Residential, Commercial, Super built-up
        """)
    
    st.warning(
        "⚠️ **Disclaimer:** This prediction is based on historical data and ML models. "
        "Actual prices may vary significantly based on market conditions, property condition, "
        "amenities, and other external factors. Not for professional real estate advice."
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666; font-size: 0.9rem;'>
            🏠 Bengaluru House Price Predictor | Powered by XGBoost ML
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
