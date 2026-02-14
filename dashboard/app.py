import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import time
from utils import InferenceEngine, PacingController
import os
import joblib # Added joblib for model config loading

st.set_page_config(page_title="RTB Optimization Dashboard", layout="wide")

st.title("Real-Time Bidding (RTB) Optimization Engine")

# --- Sidebar Configuration ---
st.sidebar.header("Campaign Settings")

# 1. Data Upload
uploaded_file = st.sidebar.file_uploader("Upload Data (Parquet/CSV)", type=["parquet", "csv"])

budget = st.sidebar.number_input("Total Budget (CPM)", min_value=1000, value=10000, step=1000)
base_bid = st.sidebar.slider("Base Bid Price", 0, 300, 100)
strategy = st.sidebar.selectbox("Bidding Strategy", ["Linear (CTR)", "Cost-Signal (CTR*CVR)", "Random"])
use_pacing = st.sidebar.checkbox("Enable Budget Pacing", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Speed")
speed = st.sidebar.slider("Speed (ms delay)", 10, 1000, 100)

col_btn1, col_btn2 = st.sidebar.columns(2)
start_btn = col_btn1.button("Start Simulation")
reset_btn = col_btn2.button("Reset Stats")

if reset_btn:
    st.session_state.indices = 0
    st.session_state.metrics = {
        "impressions": 0, "wins": 0, "clicks": 0, "conversions": 0, "spend": 0
    }
    st.session_state.history = []
    st.session_state.running = False
    st.rerun()

# --- Session State ---
if "data" not in st.session_state:
    st.session_state.data = None
    st.session_state.engine = None
    st.session_state.indices = 0
    st.session_state.metrics = {
        "impressions": 0, "wins": 0, "clicks": 0, "conversions": 0, "spend": 0
    }
    st.session_state.history = []

# --- Load Data & Model ---
@st.cache_resource
def load_resources(uploaded_file=None):
    # Resolve path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # New Calibrated Model Paths
    ctr_model_path = os.path.join(current_dir, "../ctr_model_calibrated.pkl")
    hasher_path = os.path.join(current_dir, "../feature_hasher.pkl")
    config_path = os.path.join(current_dir, "../feature_config.pkl")
    
    # New CVR Model Paths
    cvr_model_path = os.path.join(current_dir, "../cvr_model_calibrated.pkl")
    hasher_cvr_path = os.path.join(current_dir, "../feature_hasher_cvr.pkl")
    config_cvr_path = os.path.join(current_dir, "../feature_config_cvr.pkl")
    
    try:
        if uploaded_file is not None:
             if uploaded_file.name.endswith('.csv'):
                 df_full = pl.read_csv(uploaded_file)
             else:
                 df_full = pl.read_parquet(uploaded_file)
        else:
            data_path = os.path.join(current_dir, "../merged_training_data.parquet")
            # Load larger sample
            df_full = pl.read_parquet(data_path)
        
        # DEMO MODE: Ensure we have clicks to show! (Only if using default data or large enough upload)
        # Filter for clicks and some non-clicks
        if "click" in df_full.columns:
            clicks = df_full.filter(pl.col("click") == 1)
            # Get random non-clicks (e.g. 9000)
            non_clicks = df_full.filter(pl.col("click") == 0).sample(n=min(10000, len(df_full)), seed=42)
            
            # Combine
            df = pl.concat([clicks, non_clicks])
            # Shuffle
            df = df.sample(fraction=1.0, shuffle=True, seed=42)
        else:
            df = df_full.head(10000) # If no click column, just take a sample
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
        
    # Initialize Engine
    try:
        engine = InferenceEngine(
            ctr_model_path=ctr_model_path, 
            cvr_model_path=cvr_model_path,
            hasher_path=hasher_path,
            hasher_cvr_path=hasher_cvr_path,
            config_path=config_path,
            config_cvr_path=config_cvr_path
        )
    except Exception as e:
         st.error(f"Error loading engine: {e}")
         return None, None
         
    return df, engine

if st.session_state.data is None or uploaded_file is not None:
    # Reload if data is None OR if user just uploaded a file (naive check, improves with ID)
    with st.spinner("Loading Data and Models..."):
        df, engine = load_resources(uploaded_file)
        if df is not None:
            st.session_state.data = df
            st.session_state.engine = engine
            st.success("System Ready!")

# --- Main Dashboard Structure ---
tab1, tab2, tab3 = st.tabs(["🚀 Simulation", "📊 EDA Analysis", "📈 Model Performance"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    metric_ph1 = col1.empty()
    metric_ph2 = col2.empty()
    metric_ph3 = col3.empty()
    metric_ph4 = col4.empty()

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    chart_ph1 = chart_col1.empty() # Bid/Win Price
    chart_ph2 = chart_col2.empty() # CTR History
    
    status_text = st.empty()
    prediction_text = st.empty() # Placeholder for live prediction values

with tab2:
    st.header("📊 Exploratory Data Analysis")
    
    if st.session_state.data is not None:
        eda_df = st.session_state.data
        
        # --- Metrics ---
        st.subheader("Dataset Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples", f"{len(eda_df):,}")
        
        if "payprice" in eda_df.columns:
            m2.metric("Avg. Market Price", f"{eda_df['payprice'].mean():.2f} CPM")
        
        if "click" in eda_df.columns:
            ctr = eda_df["click"].mean()
            m3.metric("Overall CTR", f"{ctr:.4%}")
            
            clicks = eda_df["click"].sum()
            if clicks > 0:
                ratio = (len(eda_df) - clicks) / clicks
                m4.metric("Neg/Pos Ratio", f"{ratio:.1f}:1")
        
        st.divider()
        
        # --- 1. Price Distribution ---
        st.subheader("1. Market Price Distribution")
        st.markdown("""
        **What this shows:** The distribution of winning bid prices (market price) in the dataset.
        **Why it matters:** Understanding the market price helps in setting the **Base Bid**.
        *   If correlations are low, a static bid might work.
        *   If variance is high, dynamic bidding is essential.
        """)
        if "payprice" in eda_df.columns:
            fig_price = px.histogram(eda_df.to_pandas(), x="payprice", nbins=50, 
                                   title="Market Price Distribution", color_discrete_sequence=['#636EFA'])
            fig_price.update_layout(xaxis_title="Price (CPM)", yaxis_title="Frequency")
            st.plotly_chart(fig_price, use_container_width=True)

        # --- 2. Hourly CTR ---
        st.subheader("2. CTR by Hour of Day")
        st.markdown("""
        **What this shows:** How user engagement (Click-Through Rate) varies throughout the day.
        **Why it matters:** Use this to Implement **Time-Parting** (bidding higher during peak hours).
        """)
        if "1" in eda_df.columns and "click" in eda_df.columns:
            try:
                # Extract Hour (Assuming YYYYMMDDHHMMSS format)
                # We utilize Polars for fast string slicing
                hourly_df = (eda_df.lazy()
                            .with_columns(pl.col("1").cast(pl.Utf8).str.slice(8, 2).alias("hour"))
                            .group_by("hour")
                            .agg(pl.col("click").mean().alias("ctr"))
                            .sort("hour")
                            .collect())
                
                fig_hour = px.line(hourly_df.to_pandas(), x="hour", y="ctr", markers=True,
                                 title="CTR Trends by Hour", color_discrete_sequence=['#EF553B'])
                fig_hour.update_layout(xaxis_title="Hour of Day", yaxis_title="CTR")
                st.plotly_chart(fig_hour, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not parse timestamp/hour (Column '1') for hourly analysis. Error: {e}")

        # --- 3. Price vs Clicks ---
        st.subheader("3. Price vs. Relevance (Clicks)")
        st.markdown("""
        **What this shows:** Do expensive impressions lead to more clicks?
        **Why it matters:**
        *   If **Clicked** impressions cost significantly more, it means the market is efficient (high value = high price).
        *   If costs are similar, there are "cheap wins" available.
        """)
        if "payprice" in eda_df.columns and "click" in eda_df.columns:
             pdf = eda_df.select(["payprice", "click"]).to_pandas()
             pdf["Is Click"] = pdf["click"].map({1: "Yes (Click)", 0: "No (Non-Click)"})
             
             fig_box = px.box(pdf, x="Is Click", y="payprice", color="Is Click",
                            title="Market Price Distribution: Clicks vs Non-Clicks",
                            color_discrete_map={"Yes (Click)": "#00CC96", "No (Non-Click)": "#EF553B"})
             st.plotly_chart(fig_box, use_container_width=True)
             
    else:
        st.info("👆 Please upload a dataset or use the default one to see the analysis.")

with tab3:
    st.header("Model Performance Metrics")
    
    c1, c2 = st.columns(2)
    
    # Load Configs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../feature_config.pkl")
    config_cvr_path = os.path.join(current_dir, "../feature_config_cvr.pkl")
    
    with c1:
        st.subheader("CTR Model")
        try:
            ctr_conf = joblib.load(config_path)
            metrics = ctr_conf.get('metrics', {})
            st.metric("ROC-AUC", f"{metrics.get('auc', 0):.4f}")
            st.metric("Log Loss", f"{metrics.get('logloss', 0):.4f}")
            st.write(f"**N Features**: {ctr_conf.get('n_features', 0):,}")
            st.write("**Top Features**: Hour, UserAgent, Region") # Hardcoded for now based on training
        except Exception as e:
            st.error(f"CTR Config not found or error loading: {e}")
            
    with c2:
        st.subheader("CVR Model")
        try:
            cvr_conf = joblib.load(config_cvr_path)
            metrics = cvr_conf.get('metrics', {})
            st.metric("ROC-AUC", f"{metrics.get('auc', 0):.4f}")
            st.metric("Log Loss", f"{metrics.get('logloss', 0):.4f}")
            st.write(f"**N Features**: {cvr_conf.get('n_features', 0):,}")
        except Exception as e:
            st.error(f"CVR Config not found or error loading: {e}")

def run_simulation():
    df = st.session_state.data
    engine = st.session_state.engine
    
    if df is None or engine is None:
        st.warning("Data or Engine not loaded. Cannot run simulation.")
        return

    # Create placeholders for live updates to avoid full reruns
    progress_bar = st.progress(0)

    BATCH_SIZE = 1 # Update every step or batch? Step is better for "Real-time" feel
    
    status_text.text("Simulation Running...")
    # prediction_text is already defined in tab1, no need to redefine here.

    pacing_controller = PacingController(budget, len(df))
    # Fast-forward controller if resuming (approximation)
    if st.session_state.indices > 0:
        pacing_controller.actual_spend = st.session_state.metrics["spend"]
        # In PID, we don't need 'remaining_steps' state, it uses current index.

    
    for i in range(st.session_state.indices, len(df)):
        if not st.session_state.get("running", False):
            break
            
        row = df.row(i, named=True)
        
        # 1. Inference
        # p_cvr_cond: P(Conversion|Click) -> High (~6%)
        # p_cvr_uncond: P(Conversion) = P(Click) * P(Conversion|Click) -> Low (~0.001%)
        p_ctr, p_cvr_cond, p_cvr_uncond = engine.predict(row)
        
        # 2. Bidding
        bid_price = 0
        if strategy == "Linear (CTR)":
            bid_price = engine.bid(p_ctr, base_bid)
        elif strategy == "Cost-Signal (CTR*CVR)":
             # CALIBRATED BIDDING FORMULA (Zhang et al. 2014)
             # Bid = BaseBid * (pCTR / AvgCTR) * (pCVR_cond / AvgCVR_cond)
             
             avg_ctr = 0.0002
             avg_cvr_cond = 0.063
             
             # Combined factor
             # Note: p_cvr_cond is used because calibration avg is conditional
             bid_price = int(base_bid * (p_ctr / avg_ctr) * (p_cvr_cond / avg_cvr_cond))
             
             # Safety cap
             bid_price = min(bid_price, 300) # Cap at max market price
        else:
            # Default to a random bid if strategy is not recognized or "Random"
            bid_price = np.random.randint(1, 300) # Example random bid
            
        # 2a. Apply Pacing
        if use_pacing:
            # Use current pacing factor
            factor = pacing_controller.pacing_factor
            bid_price = int(bid_price * factor)
            
        # 3. Simulate Auction (Win if bid > payprice)
        payprice = row["payprice"]
        win = bid_price > payprice
        
        if win:
            if st.session_state.metrics["spend"] + payprice <= budget:
                st.session_state.metrics["wins"] += 1
                st.session_state.metrics["spend"] += payprice
                
                if row["click"] == 1:
                    st.session_state.metrics["clicks"] += 1
                if row["conversion"] == 1:
                    st.session_state.metrics["conversions"] += 1
            else:
                st.warning("Budget Exhausted!")
                st.session_state.running = False
                break
                
        # 4. Update Pacing Controller
        if use_pacing:
            spent = payprice if win else 0
            pacing_controller.update(spent, i+1)
                
        st.session_state.metrics["impressions"] += 1
        st.session_state.indices = i + 1
        
        # Calculated Metrics
        ctr = 0
        if st.session_state.metrics["wins"] > 0:
            ctr = st.session_state.metrics["clicks"] / st.session_state.metrics["wins"]
            
        cvr = 0
        if st.session_state.metrics["clicks"] > 0:
            cvr = st.session_state.metrics["conversions"] / st.session_state.metrics["clicks"]

        # UI Updates
        metric_ph1.metric("Impressions", st.session_state.metrics["impressions"])
        metric_ph2.metric("Wins/Bids", f"{st.session_state.metrics['wins']} / {st.session_state.metrics['impressions']}")
        metric_ph3.metric("Spend", f"${st.session_state.metrics['spend']:.2f}", f"{budget - st.session_state.metrics['spend']:.2f} left")
        metric_ph4.metric("CTR (Realized)", f"{ctr:.4%}", f"CVR: {cvr:.4%}")
        
        # History for charts
        st.session_state.history.append({
            "step": i,
            "bid": bid_price,
            "win_price": payprice if win else 0, # 0 if lost? or just payprice? payprice is market price.
            "market_price": payprice,
            "ctr": ctr,
            "is_win": win
        })
        
        # Update Charts every 50 steps to not kill performance
        if i % 20 == 0:
            hist_df = pd.DataFrame(st.session_state.history[-200:]) # Last 200 points
            
            # Chart 1: Bidding Stream
            fig1 = px.line(hist_df, x="step", y=["bid", "market_price"], 
                           title="Real-time Bidding Stream", 
                           color_discrete_sequence=["#00CC96", "#EF553B"])
            chart_ph1.plotly_chart(fig1, use_container_width=True)
            
            # Chart 2: CTR Trend
            if len(hist_df) > 1:
                fig2 = px.line(hist_df, x="step", y="ctr", title="Realized CTR Trend")
                chart_ph2.plotly_chart(fig2, use_container_width=True)

        # Live Prediction Feed (Every 10 steps to be readable)
        if i % 10 == 0:
            prediction_text.markdown(
                f"**Live Prediction**: pCTR=`{p_ctr:.2%}` | "
                f"pCVR(Cond)=`{p_cvr_cond:.1%}` | "
                f"**pCVR(True)**=`{p_cvr_uncond:.5%}`"
            )

        # Progress bar
        progress = min(1.0, (i + 1) / len(df))
        progress_bar.progress(progress)
        
        time.sleep(speed / 1000)

if start_btn:
    st.session_state.running = True
    run_simulation()
    

