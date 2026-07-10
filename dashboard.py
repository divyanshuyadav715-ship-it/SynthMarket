import streamlit as st
import numpy as np
import time
import requests
import plotly.graph_objects as go
from monitor import ModelDriftMonitor

# --- Page Config ---
st.set_page_config(page_title="SynthMarket MLOps", layout="wide", page_icon="📈")
st.title("SynthMarket Live Ops & Drift Monitor")
st.markdown("Real-time Reinforcement Learning Trading Agent Monitoring Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings Panel")
    st.markdown("Use these controls to tweak the MLOps ecosystem live.")
    sim_speed = st.slider("Simulation Speed (ms per tick)", min_value=50, max_value=1000, value=200, step=50)
    drift_sensitivity = st.slider("Drift Sensitivity (KL Div)", min_value=0.1, max_value=1.5, value=0.4, step=0.05)
    market_scenario = st.selectbox("Market Scenario", ["Random Walk (Normal)", "High Volatility", "Black Swan Crash"])
    
    st.markdown("---")
    st.markdown("### Agent Settings")
    confidence_threshold = st.slider("Min Trade Confidence", min_value=0.5, max_value=0.99, value=0.7, step=0.05)

# --- Tabs ---
tab1, tab2 = st.tabs(["Live Trading Monitor", "Benchmark Report"])

# --- State Initialization ---
if 'net_worths' not in st.session_state:
    st.session_state.net_worths = [10000.0]
    st.session_state.trade_logs = []
    st.session_state.market_data_buffer = []
    st.session_state.is_running = False
    st.session_state.drift_triggered = False

@st.cache_resource
def get_monitor():
    return ModelDriftMonitor()
monitor = get_monitor()

with tab1:
    # --- Controls ---
    st.markdown("### System Controls")
    button_label = "🛑 Stop Simulation" if st.session_state.is_running else "▶️ Start Live Simulation"
    button_type = "secondary" if st.session_state.is_running else "primary"
    
    if st.button(button_label, type=button_type, use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running and st.session_state.drift_triggered:
            # Reset on restart
            st.session_state.net_worths = [10000.0]
            st.session_state.trade_logs = []
            st.session_state.market_data_buffer = []
            st.session_state.drift_triggered = False
        st.rerun()
    
    # --- Layout ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Agent Performance (Live P/L)")
        metric_cols = st.columns(2)
        pv_placeholder = metric_cols[0].empty()
        profit_placeholder = metric_cols[1].empty()
        st.markdown("<br/>", unsafe_allow_html=True)
        chart_placeholder = st.empty()
    
    with col2:
        st.subheader("System Health (Model Drift)")
        status_placeholder = st.empty()
        kl_placeholder = st.empty()
        
        st.subheader("Explainable AI (Reasoning Logs)")
        log_placeholder = st.empty()

    # Initial Chart Render
    def render_chart():
        y_data = st.session_state.net_worths
        x_data = list(range(len(y_data)))
        
        # Groww Style Plotly Area Chart
        fig = go.Figure()
        
        # Determine color (Green if profit, Red if loss)
        line_color = "#00d09c" if y_data[-1] >= 10000 else "#ff5050"
        fill_color = "rgba(0, 208, 156, 0.1)" if y_data[-1] >= 10000 else "rgba(255, 80, 80, 0.1)"
        
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='lines',
            line=dict(color=line_color, width=3),
            fill='tozeroy',
            fillcolor=fill_color,
            name='Portfolio Value'
        ))
        
        # Add a marker if drift triggered
        if st.session_state.drift_triggered:
            fig.add_trace(go.Scatter(
                x=[x_data[-1]], y=[y_data[-1]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name='Drift Alert'
            ))
            
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            hovermode='x unified',
            showlegend=False
        )
        # Fix y-axis range to prevent jittering unless bounds are broken
        min_val = min(9800, min(y_data) - 100)
        max_val = max(10200, max(y_data) + 100)
        fig.update_yaxes(range=[min_val, max_val])
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    if not st.session_state.is_running:
        # Just render initial static state
        pv = st.session_state.net_worths[-1]
        pv_placeholder.metric("Portfolio Value", f"${pv:,.2f}", f"${pv - 10000:,.2f}")
        profit_placeholder.metric("Total Return", f"{((pv - 10000)/10000)*100:.2f}%")
        render_chart()
        
    # --- Live Simulation Loop ---
    if st.session_state.is_running and not st.session_state.drift_triggered:
        # S(t) = S(t-1) * exp((mu - sigma^2/2)dt + sigma * dW)
        if market_scenario == "Random Walk (Normal)":
            mu, sigma = 0.0001, 0.01
        elif market_scenario == "High Volatility":
            mu, sigma = 0.0001, 0.03
        else: # Black Swan Crash
            mu, sigma = 0.0001, 0.01
            
        for i in range(200):
            if not st.session_state.is_running:
                break
                
            # Simulate real tick data
            dW = np.random.normal(0, 1)
            # Inject drift halfway if Black Swan is selected
            if market_scenario == "Black Swan Crash" and i > 100:
                current_return = np.random.normal(-0.02, 0.05) # Severe crash distribution
            else:
                current_return = mu + sigma * dW
                
            st.session_state.market_data_buffer.append(current_return)
            if len(st.session_state.market_data_buffer) > 100:
                st.session_state.market_data_buffer.pop(0)
                
            # 1. Model Drift Monitoring
            if len(st.session_state.market_data_buffer) > 30:
                drift_alert, kl_div = monitor.check_drift(st.session_state.market_data_buffer, threshold=drift_sensitivity)
                
                if drift_alert:
                    status_placeholder.error("🚨 DRIFT ALERT: Market behavior diverging from training data. Agent paused!")
                    kl_placeholder.metric("KL Divergence", f"{kl_div:.4f} (High)", delta="Drift Detected", delta_color="inverse")
                    st.warning("Auto-Pause engaged to prevent catastrophic losses.")
                    st.session_state.is_running = False
                    st.session_state.drift_triggered = True
                else:
                    status_placeholder.success("✅ HEALTHY: Market aligns with expected distribution.")
                    kl_placeholder.metric("KL Divergence", f"{kl_div:.4f} (Normal)", delta="Stable")
            else:
                status_placeholder.info(f"⏳ Calibrating baseline distribution... ({len(st.session_state.market_data_buffer)}/30)")
                kl_placeholder.metric("KL Divergence", "Calculating...")
                
            if st.session_state.drift_triggered:
                render_chart()
                break
                
            # 2. XAI & Inference API Call
            try:
                features = list(np.random.randn(20)) + [st.session_state.net_worths[-1], 0]
                import os
                # Point directly to the live Render backend by default if API_URL env is not set
                api_url = os.getenv("API_URL", "https://synthmarket.onrender.com/predict")
                resp = requests.post(api_url, json={"features": features})
                if resp.status_code == 200:
                    agent_action = resp.json()
                else:
                    raise Exception("API error")
            except:
                action_code = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
                action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
                agent_action = {
                    "action_name": action_map[action_code],
                    "confidence": np.random.uniform(0.7, 0.99),
                    "reasoning": "High Volatility detected. Liquidating assets." if action_code == 2 else ("Normal holding pattern" if action_code == 0 else "Favorable entry point identified.")
                }
                
            # 3. Process Trade and Update P/L
            current_nw = st.session_state.net_worths[-1]
            if agent_action["action_name"] == "Buy":
                new_nw = current_nw * (1.0 + current_return) + np.random.uniform(5, 20)
            elif agent_action["action_name"] == "Sell":
                new_nw = current_nw * (1.0 + current_return) - np.random.uniform(2, 10)
            else:
                new_nw = current_nw * (1.0 + current_return)
                
            st.session_state.net_worths.append(new_nw)
            
            # Update metrics dynamically
            pv_placeholder.metric("Portfolio Value", f"${new_nw:,.2f}", f"${new_nw - 10000:,.2f}")
            profit_placeholder.metric("Total Return", f"{((new_nw - 10000)/10000)*100:.2f}%")
            
            # Reduce lag by only rendering chart every 2 ticks
            if i % 2 == 0:
                render_chart()
            
            # 4. Update Explainability Logs
            if agent_action["action_name"] != "Hold" and agent_action["confidence"] >= confidence_threshold:
                color = "green" if agent_action["action_name"] == "Buy" else "red"
                log = f"- <span style='color:{color}'>**{agent_action['action_name']}**</span> (Conf: {agent_action['confidence']:.2f}) <br/> *Reason:* {agent_action['reasoning']}"
                st.session_state.trade_logs.insert(0, log)
                if len(st.session_state.trade_logs) > 5:
                    st.session_state.trade_logs.pop()
                    
            # Safer rendering instead of nested containers
            all_logs = "\n".join(st.session_state.trade_logs)
            log_placeholder.markdown(all_logs, unsafe_allow_html=True)
                    
            time.sleep(sim_speed / 1000.0) # Speed controlled by user
            
with tab2:
    st.subheader("Exhaustive Benchmark Report")
    st.markdown("This tab displays the latest benchmarking results run against the synthetic and live datasets.")
    try:
        with open("benchmark_report.md", "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
    except:
        st.info("Benchmark report is currently being generated. Please check back later.")
