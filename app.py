import streamlit as st
import numpy as np
import pandas as pd
import time
import end 

st.set_page_config(page_title="Coevolutionary Recommender", layout="wide")
st.title("Adaptive Recommendation Engine")


@st.cache_data
def load_and_prep_data():
    matrix, users, items = end.get_rating_matrix()
    train, test = end.train_test_split_matrix(matrix)
    return matrix, train, test, users, items

with st.spinner('Loading MovieLens Dataset...'):
    matrix, train, test, users, items = load_and_prep_data()

st.sidebar.success(f"Loaded {len(users)} users and {len(items)} items.")

st.sidebar.header("EA Parameters")
strategy = st.sidebar.selectbox("Coevolution Strategy", ("Cooperative", "Competitive"))
generations = st.sidebar.slider("Generations", min_value=10, max_value=200, value=50)

target_user = st.sidebar.selectbox("Target User ID for Recommendations", users)

if st.sidebar.button("Run Coevolution Training", type="primary"):
    
    st.subheader("Training Convergence (RMSE)")
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    rmse_history = []
    mask = end.get_rated_mask(test)
    
    # --- THE MISSING LINK: YOUR EA ALGORITHM ---
    # Right now, you have the data (train/test) and the fitness function (end.calculate_fitness).
    # You are missing the actual Coevolutionary Algorithm that evolves latent vectors for users/items.
    # The loop below is still a simulation. Your backend team needs to provide a function that 
    # executes a generation of the EA and returns the updated predicted matrix.
    
    for gen in range(generations):
        time.sleep(0.05) # Simulating EA processing time
        
        # MOCK EA STEP: Simulating the EA getting better at predicting the matrix over time.
        # Replace this block with your actual EA generation step.
        noise_level = max(0.5, 3.0 - (2.5 * (gen / generations)))
        mock_predicted_matrix = train + np.random.normal(0, noise_level, size=train.shape)
        mock_predicted_matrix = np.clip(mock_predicted_matrix, 1, 5)
        
        current_rmse = end.calculate_fitness(mock_predicted_matrix, test, mask=mask)
        rmse_history.append(current_rmse)
        
        chart_placeholder.line_chart(rmse_history, height=300)
        progress_bar.progress((gen + 1) / generations)
        
    st.success(f"Training complete. Final RMSE: {rmse_history[-1]:.4f}")
    
    st.subheader(f"Top Recommendations for User {target_user}")
    
    # MOCK OUTPUT: Once your EA is done, it will yield predicted ratings for unrated items.
    # You will sort those predictions and display the top N items here.
    mock_results = pd.DataFrame({
        "Item ID (Movie)": np.random.choice(items, 5, replace=False),
        "Predicted Rating": [4.8, 4.5, 4.3, 4.1, 4.0]
    })
    st.dataframe(mock_results, use_container_width=True)

else:
    st.info("Adjust parameters and click 'Run' to train the model.")