"""Streamlit Dashboard for RTB DSP Optimization System"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time, os, sys, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Official Bid class from mandatory package
from com.dtu.hackathon.bidding.Bid import Bid as OfficialBid

# Supporting modules
from src.data_loader import load_data
from src.ipinyou_loader import load_ipinyou_data
from src.features import FeatureEngineer
from src.models import CTRModel
from src.bidder import ConstantBidding, RandomBidding, MCPCBidding, LinearBidding
from src.simulator import AuctionSimulator
from src.config import *

st.set_page_config(page_title="RTB DSP Optimizer", page_icon="💰", layout="wide")

st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: bold; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 1rem 0;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;}
</style>
""", unsafe_allow_html=True)

if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'simulation_complete' not in st.session_state: st.session_state.simulation_complete = False
if 'use_ipinyou' not in st.session_state: st.session_state.use_ipinyou = False

@st.cache_data
def load_or_generate_data(regenerate=False, use_ipinyou=False):
    if use_ipinyou:
        return load_ipinyou_data(force_download=regenerate)
    return load_data('data/synthetic_rtb.csv', regenerate=regenerate)

@st.cache_resource
def train_model_pipeline(df):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    engineer = FeatureEngineer()
    aggregates = engineer.build_aggregates(train_df)
    engineer.save_aggregates()
    X_train, y_train = engineer.create_training_features(train_df), train_df['click']
    X_test, y_test = engineer.create_training_features(test_df), test_df['click']
    model = CTRModel(C=LR_C, max_iter=LR_MAX_ITER)
    train_metrics = model.train(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    model.save()
    return {'model': model, 'aggregates': aggregates, 'engineer': engineer, 'train_metrics': train_metrics,
            'test_metrics': test_metrics, 'test_predictions': test_metrics['predictions'], 'test_labels': y_test, 'test_df': test_df}

def main():
    st.markdown('<h1 class="main-header">💰 RTB DSP Optimization System</h1>', unsafe_allow_html=True)
    st.markdown("### Real-Time Bidding • Machine Learning • Auction Simulation")
    
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.markdown("### 📦 Package: `com.dtu.hackathon.bidding`")
    
    with st.sidebar.expander("📊 Data Management", expanded=True):
        data_source = st.radio("Data Source", ["Synthetic (Quick)", "iPinYou (Real)"], horizontal=True)
        st.session_state.use_ipinyou = (data_source == "iPinYou (Real)")
        
        if st.session_state.use_ipinyou:
            st.info("🔒 **Real Data Only Mode**\nNo synthetic fallback - using actual iPinYou dataset")
        
        if st.button("🔄 Generate/Download Data", width="stretch"):
            with st.spinner("Loading data..."):
                try:
                    df = load_or_generate_data(regenerate=True, use_ipinyou=st.session_state.use_ipinyou)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df):,} bid requests")
                except Exception as e:
                    st.error(f"❌ Error loading data:\n{str(e)}")
                    st.session_state.data_loaded = False
                    
        if st.button("📁 Load Cached Data", width="stretch"):
            with st.spinner("Loading data..."):
                try:
                    df = load_or_generate_data(regenerate=False, use_ipinyou=st.session_state.use_ipinyou)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df):,} bid requests")
                except Exception as e:
                    st.error(f"❌ Error loading data:\n{str(e)}")
                    st.session_state.data_loaded = False
    
    with st.sidebar.expander("🤖 Model Training", expanded=True):
        if st.button("🚀 Train CTR Model", width="stretch", disabled=not st.session_state.data_loaded):
            with st.spinner("Training logistic regression model..."):
                df = load_or_generate_data(use_ipinyou=st.session_state.use_ipinyou)
                results = train_model_pipeline(df)
                st.session_state.model_trained = True
                st.session_state.train_results = results
                st.success(f"✅ Model trained - AUC: {results['test_metrics']['auc']:.4f}")
    
    with st.sidebar.expander("🧪 Performance Tests", expanded=False):
        if st.button("🔬 Run Latency Test", width="stretch"):
            st.info("Run: `pytest tests/test_latency.py -v`")
        if st.button("💾 Run Memory Test", width="stretch"):
            st.info("Run: `pytest tests/test_memory.py -v`")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔬 EDA", "🎯 Model Performance", "⚡ Live Simulator"])
    
    with tab1:
        st.header("🎯 Project Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🎯 Objective")
            st.write("Build an end-to-end RTB bidding system with CTR Prediction, Bidding Strategies, Auction Simulation, and Budget Optimization")
        with col2:
            st.markdown("#### 🏗️ Architecture")
            st.write("**Offline**: Data → Features → CTR Model → Export Weights\n\n**Online**: O(1) lookup → CTR prediction → Bid calculation → Pacing")
        with col3:
            st.markdown("#### 📊 Metrics")
            st.write("**KPI**: Clicks + 5 × Conversions\n\n**Metrics**: AUC, Win Rate, eCPC, eCPA, Budget Utilization")
        
        st.divider()
        st.subheader("🔧 System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1: st.metric("Data", "✅ Ready" if st.session_state.data_loaded else "⏸️ Not Loaded")
        with status_col2: st.metric("Model", "✅ Trained" if st.session_state.model_trained else "⏸️ Not Trained")
        with status_col3: st.metric("Simulation", "✅ Complete" if st.session_state.simulation_complete else "⏸️ Pending")
    
    with tab2:
        st.header("🔬 Exploratory Data Analysis")
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first from the sidebar")
        else:
            df = load_or_generate_data()
            st.subheader("📊 Dataset Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Total Requests", f"{len(df):,}")
            with col2: st.metric("Clicks", f"{df['click'].sum():,}")
            with col3: st.metric("CTR", f"{df['click'].mean():.3%}")
            with col4: st.metric("Conversions", f"{df['conversion'].sum():,}")
            with col5: st.metric("CVR", f"{df['conversion'].mean():.4%}")
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                hourly_ctr = df.groupby('hour')['click'].mean().reset_index()
                fig = px.line(hourly_ctr, x='hour', y='click', title='CTR by Hour of Day', labels={'click': 'CTR'}, markers=True)
                fig.update_traces(line_color='#667eea', line_width=3)
                st.plotly_chart(fig, width="stretch")
                
                fig = px.histogram(df, x='payingprice', nbins=50, title='Market Price Distribution')
                fig.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                campaign_stats = df.groupby('campaign_id').agg({'click': ['sum', 'count', 'mean']}).reset_index()
                campaign_stats.columns = ['campaign_id', 'clicks', 'requests', 'ctr']
                campaign_stats = campaign_stats.nlargest(10, 'requests')
                fig = px.bar(campaign_stats, x='campaign_id', y='ctr', title='Top 10 Campaigns by CTR')
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, width="stretch")
                
                weekday_ctr = df.groupby('weekday')['click'].mean().reset_index()
                weekday_ctr['day_name'] = weekday_ctr['weekday'].map(lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
                fig = px.bar(weekday_ctr, x='day_name', y='click', title='CTR by Day of Week')
                fig.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig, width="stretch")
    
    with tab3:
        st.header("🎯 CTR Model Performance")
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first from the sidebar")
        else:
            results = st.session_state.train_results
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Train AUC", f"{results['train_metrics']['auc']:.4f}")
            with col2: st.metric("Test AUC", f"{results['test_metrics']['auc']:.4f}")
            with col3: st.metric("Train LogLoss", f"{results['train_metrics']['logloss']:.4f}")
            with col4: st.metric("Test LogLoss", f"{results['test_metrics']['logloss']:.4f}")
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                from sklearn.metrics import roc_curve
                y_test, y_pred = results['test_labels'], results['test_predictions']
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={results["test_metrics"]["auc"]:.4f})', line=dict(color='#667eea', width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', width=2, dash='dash')))
                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=y_pred[y_test == 1], name='Clicks', opacity=0.7, marker_color='#667eea', nbinsx=50))
                fig.add_trace(go.Histogram(x=y_pred[y_test == 0], name='No Clicks', opacity=0.7, marker_color='#764ba2', nbinsx=50))
                fig.update_layout(title='Predicted CTR Distribution', xaxis_title='Predicted CTR', yaxis_title='Count', barmode='overlay')
                st.plotly_chart(fig, width="stretch")
    
    with tab4:
        st.header("⚡ Interactive Auction Simulator")
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first")
        else:
            df = load_or_generate_data()
            with open('experiments/results/aggregates.pkl', 'rb') as f: aggregates = pickle.load(f)
            with open('experiments/results/ctr_model_weights.json', 'r') as f: weights_data = json.load(f)
            weights, bias = np.array(weights_data['weights']), weights_data['bias']
            
            st.subheader("🎛️ Simulation Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_market_spend = int(df['payingprice'].sum())
                budget_option = st.selectbox("Budget Level", ["Low (1/32)", "Medium (1/8)", "High (1/2)"])
                budget = {"Low (1/32)": total_market_spend // 32, "Medium (1/8)": total_market_spend // 8, "High (1/2)": total_market_spend // 2}[budget_option]
                st.metric("Budget", f"${budget:,}")
            with col2:
                sample_size = st.slider("Sample Size (for speed)", 1000, min(50000, len(df)), 10000, 1000)
            with col3:
                strategies_to_run = st.multiselect("Strategies to Compare", ["Constant", "Random", "MCPC", "Linear", "ML-Linear"], ["Constant", "Linear", "ML-Linear"])
            
            if st.button("🚀 Run Simulation", width="stretch"):
                bidders = {}
                if "Constant" in strategies_to_run: bidders["Constant"] = ConstantBidding(bid_price=80)
                if "Random" in strategies_to_run: bidders["Random"] = RandomBidding(min_bid=30, max_bid=150)
                if "MCPC" in strategies_to_run: bidders["MCPC"] = MCPCBidding(max_ecpc=1000)
                if "Linear" in strategies_to_run: bidders["Linear"] = LinearBidding(base_bid=BASE_BID, avg_ctr=aggregates['global_ctr'])
                if "ML-Linear" in strategies_to_run: bidders["ML-Linear (Official)"] = OfficialBid(model_path='experiments/results', conversion_weight=5)
                
                sample_df = df.sample(n=sample_size, random_state=42)
                progress_bar, status_text, results_list = st.progress(0), st.empty(), []
                
                for i, (name, bidder) in enumerate(bidders.items()):
                    status_text.text(f"Running {name} strategy...")
                    sim = AuctionSimulator(bidder, budget, name)
                    results_list.append(sim.simulate(sample_df, verbose=False))
                    progress_bar.progress((i + 1) / len(bidders))
                
                status_text.text("✅ Simulation complete!")
                time.sleep(0.5)
                progress_bar.empty(); status_text.empty()
                
                results_df = pd.DataFrame(results_list)
                st.session_state.simulation_results, st.session_state.simulation_complete = results_df, True
                
                st.subheader("📊 Simulation Results")
                metric_cols = st.columns(len(bidders))
                for i, (idx, row) in enumerate(results_df.iterrows()):
                    with metric_cols[i]:
                        st.markdown(f"### {row['strategy']}")
                        st.metric("KPI", f"{row['kpi']:.1f}")
                        st.metric("Clicks", f"{row['clicks']}")
                        st.metric("Spent", f"${row['spent']:,}")
                        st.metric("Win Rate", f"{row['win_rate']:.2%}")
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(results_df, x='strategy', y='kpi', title='KPI Comparison (Clicks + 5×Conversions)', color='kpi', color_continuous_scale='viridis', text='kpi')
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                
                with col2:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=['Win Rate', 'CTR'])
                    fig.add_trace(go.Bar(x=results_df['strategy'], y=results_df['win_rate'], name='Win Rate', marker_color='#667eea'), row=1, col=1)
                    fig.add_trace(go.Bar(x=results_df['strategy'], y=results_df['ctr'], name='CTR', marker_color='#764ba2'), row=1, col=2)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                
                best_strategy = results_df.loc[results_df['kpi'].idxmax()]
                st.success(f"### 🏆 Best Performing Strategy: **{best_strategy['strategy']}**\n\n- **KPI**: {best_strategy['kpi']:.1f}\n- **Clicks**: {best_strategy['clicks']}\n- **Win Rate**: {best_strategy['win_rate']:.2%}")

if __name__ == "__main__":
    main()
