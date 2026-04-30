import streamlit as st
import numpy as np
import pandas as pd
import random
import data
import coevolution_base as cb


def run_ea(train, test, generations, seed):
    n_users, n_items = train.shape
    latent_dim = cb.GENOME_LENGTH
    
    random.seed(seed)
    
    users = [cb.create_individual() for _ in range(n_users)]
    items = [cb.create_individual() for _ in range(n_items)]
    
    rmse_history = []
    
    for gen in range(generations):
        user_vectors = np.array(users)
        item_vectors = np.array(items)
        
        predicted = user_vectors @ item_vectors.T
        predicted = predicted * 4 + 1
        predicted = np.clip(predicted, 1, 5)
        
        train_mask = train > 0
        test_mask = test > 0
        
        if train_mask.sum() > 0:
            train_rmse = data.calculate_fitness(predicted, train, mask=train_mask)
        else:
            train_rmse = 0.0
        
        if test_mask.sum() > 0:
            test_rmse = data.calculate_fitness(predicted, test, mask=test_mask)
        else:
            test_rmse = train_rmse
        
        rmse_history.append(test_rmse)
        
        user_fitnesses = []
        for i in range(n_users):
            user_pred = predicted[i, :]
            user_actual = train[i, :]
            user_mask = user_actual > 0
            if user_mask.sum() > 0:
                user_rmse = data.calculate_fitness(user_pred, user_actual, mask=user_mask)
            else:
                user_rmse = 0.5
            user_fitnesses.append(1.0 / (user_rmse + 1e-6))
        
        item_fitnesses = []
        for j in range(n_items):
            item_pred = predicted[:, j]
            item_actual = train[:, j]
            item_mask = item_actual > 0
            if item_mask.sum() > 0:
                item_rmse = data.calculate_fitness(item_pred, item_actual, mask=item_mask)
            else:
                item_rmse = 0.5
            item_fitnesses.append(1.0 / (item_rmse + 1e-6))
        
        users = cb.next_generation(users, user_fitnesses)
        items = cb.next_generation(items, item_fitnesses)
    
    final_user_vectors = np.array(users)
    final_item_vectors = np.array(items)
    final_predicted = final_user_vectors @ final_item_vectors.T
    final_predicted = final_predicted * 4 + 1
    final_predicted = np.clip(final_predicted, 1, 5)
    
    return final_predicted, rmse_history


st.set_page_config(page_title="Coevolutionary Recommender", layout="wide")
st.title("Adaptive Recommendation Engine")


@st.cache_data
def load_and_prep_data():
    matrix, users, items, titles = data.get_rating_matrix()
    train, test = data.train_test_split_matrix(matrix)
    return matrix, train, test, users, items, titles

with st.spinner('Loading MovieLens Dataset...'):
    matrix, train, test, users, items, titles = load_and_prep_data()

st.sidebar.success(f"Loaded {len(users)} users and {len(items)} items.")

mean_rating = matrix[matrix > 0].mean()
naive_pred = np.full_like(test, mean_rating)
mask = test > 0
baseline_rmse = data.calculate_fitness(naive_pred, test, mask=mask)

st.sidebar.info(f"**Baseline RMSE**: {baseline_rmse:.4f}")

st.sidebar.header("EA Parameters")
generations = st.sidebar.slider("Generations", min_value=10, max_value=200, value=50)

target_user = st.sidebar.selectbox("Target User ID for Recommendations", users)

user_idx = np.where(users == target_user)[0][0]
user_ratings = matrix[user_idx]
five_star_mask = user_ratings == 5

st.sidebar.subheader("Your 5-Star Movies")
if five_star_mask.sum() > 0:
    five_star_items = items[five_star_mask]
    five_star_titles = [titles.get(iid, f"Item {iid}") for iid in five_star_items[:5]]
    for title in five_star_titles:
        st.sidebar.write(f"⭐ {title}")
    if len(five_star_items) > 5:
        st.sidebar.caption(f"+{len(five_star_items) - 5} more")
else:
    st.sidebar.caption("No 5-star ratings for this user")

if st.sidebar.button("Run Coevolution Training", type="primary"):
    
    st.subheader("Training Convergence (RMSE)")
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    rmse_history = []
    
    predicted_matrix, rmse_history = run_ea(
        train=train,
        test=test,
        generations=generations,
        seed=42
    )
    
    for gen, rmse in enumerate(rmse_history):
        rmse_history_partial = rmse_history[:gen+1]
        chart_placeholder.line_chart(rmse_history_partial, height=300)
        progress_bar.progress((gen + 1) / generations)
    
    st.success(f"Training complete. Final RMSE: {rmse_history[-1]:.4f}")
    
    st.subheader(f"Top Recommendations for User {target_user}")
    
    unrated_mask = user_ratings == 0
    
    predicted_ratings = predicted_matrix[user_idx]
    predicted_ratings_clipped = np.clip(predicted_ratings, 1, 5)
    
    unrated_indices = np.where(unrated_mask)[0]
    unrated_predictions = predicted_ratings_clipped[unrated_indices]
    
    if len(unrated_indices) > 0:
        top_n_indices = unrated_indices[np.argsort(unrated_predictions)[::-1][:5]]
        
        recommendations = []
        for idx in top_n_indices:
            item_id = items[idx]
            movie_title = titles.get(item_id, f"Unknown Movie (ID: {item_id})")
            pred_rating = predicted_ratings_clipped[idx]
            recommendations.append({
                "Movie Title": movie_title,
                "Item ID": int(item_id),
                "Predicted Rating": float(pred_rating)
            })
        
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("This user has rated all items.")
    
    st.subheader("Training Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial RMSE", f"{rmse_history[0]:.4f}")
    col2.metric("Final RMSE", f"{rmse_history[-1]:.4f}")
    col3.metric("Best Improvement", f"{max(rmse_history) - rmse_history[-1]:.4f}")

else:
    st.info("Adjust parameters and click 'Run' to train the model.")
