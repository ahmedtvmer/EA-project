import streamlit as st
import numpy as np
import pandas as pd
import data
import coevolution_engine as ce


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
        st.sidebar.write(f" {title}")
    if len(five_star_items) > 5:
        st.sidebar.caption(f"+{len(five_star_items) - 5} more")
else:
    st.sidebar.caption("No 5-star ratings for this user")

if st.sidebar.button("Run Coevolution Training", type="primary"):
    
    st.subheader("Training Convergence (RMSE)")
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner("Running both strategies for comparison..."):
        coop_user_pop, coop_item_pop, coop_history = ce.run_coevolution(
            train_matrix=train,
            test_matrix=test,
            n_generations=generations,
            strategy="cooperative",
            seed=42,
            verbose=False
        )
        
        progress_bar.progress(50)
        
        comp_user_pop, comp_item_pop, comp_history = ce.run_coevolution(
            train_matrix=train,
            test_matrix=test,
            n_generations=generations,
            strategy="competitive",
            seed=42,
            verbose=False
        )
        
        progress_bar.progress(100)
    
    coop_rmse = [h["test_rmse"] for h in coop_history]
    comp_rmse = [h["test_rmse"] for h in comp_history]
    
    comparison_df = pd.DataFrame({
        "Cooperative": coop_rmse,
        "Competitive": comp_rmse,
    })
    chart_placeholder.line_chart(comparison_df, height=300)
    
    coop_final = coop_history[-1]["test_rmse"]
    comp_final = comp_history[-1]["test_rmse"]
    
    st.success(f"Training complete. Cooperative RMSE: {coop_final:.4f} | Competitive RMSE: {comp_final:.4f}")
    
    st.subheader("Strategy Comparison")
    
    comparison_table = pd.DataFrame([
        {
            "Strategy": "Cooperative",
            "Initial RMSE": round(coop_history[0]["test_rmse"], 4),
            "Final RMSE": round(coop_final, 4),
            "Improvement": round(baseline_rmse - coop_final, 4),
            "Avg User Fitness": round(coop_history[-1]["avg_user_fitness"], 4),
            "Avg Item Fitness": round(coop_history[-1]["avg_item_fitness"], 4),
        },
        {
            "Strategy": "Competitive",
            "Initial RMSE": round(comp_history[0]["test_rmse"], 4),
            "Final RMSE": round(comp_final, 4),
            "Improvement": round(baseline_rmse - comp_final, 4),
            "Avg User Fitness": round(comp_history[-1]["avg_user_fitness"], 4),
            "Avg Item Fitness": round(comp_history[-1]["avg_item_fitness"], 4),
        },
    ])
    st.dataframe(comparison_table, width="stretch", hide_index=True)
    
    st.subheader(f"Top Recommendations for User {target_user}")
    
    rated_mask = user_ratings > 0
    
    coop_results = ce.recommend(user_idx, coop_user_pop, coop_item_pop, items, titles, rated_mask, top_n=5)
    comp_results = ce.recommend(user_idx, comp_user_pop, comp_item_pop, items, titles, rated_mask, top_n=5)
    
    col_coop, col_comp = st.columns(2)
    
    with col_coop:
        st.markdown("### Cooperative")
        if coop_results:
            coop_df = pd.DataFrame(coop_results)
            coop_df = coop_df.rename(columns={"title": "Movie Title", "item_id": "Item ID", "predicted_rating": "Predicted Rating"})
            st.dataframe(coop_df, width="stretch", hide_index=True)
        else:
            st.info("This user has rated all items.")
    
    with col_comp:
        st.markdown("### Competitive")
        if comp_results:
            comp_df = pd.DataFrame(comp_results)
            comp_df = comp_df.rename(columns={"title": "Movie Title", "item_id": "Item ID", "predicted_rating": "Predicted Rating"})
            st.dataframe(comp_df, width="stretch", hide_index=True)
        else:
            st.info("This user has rated all items.")
    
    st.subheader("Training Statistics")
    
    coop_best = max(h["test_rmse"] for h in coop_history)
    comp_best = max(h["test_rmse"] for h in comp_history)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Coop Initial RMSE", f"{coop_history[0]['test_rmse']:.4f}")
    col2.metric("Coop Final RMSE", f"{coop_history[-1]['test_rmse']:.4f}")
    col3.metric("Coop Best Improvement", f"{coop_best - coop_history[-1]['test_rmse']:.4f}")
    col4.metric("Comp Initial RMSE", f"{comp_history[0]['test_rmse']:.4f}")
    col5.metric("Comp Final RMSE", f"{comp_history[-1]['test_rmse']:.4f}")
    col6.metric("Comp Best Improvement", f"{comp_best - comp_history[-1]['test_rmse']:.4f}")

else:
    st.info("Adjust parameters and click 'Run' to train the model.")
