import streamlit as st
import numpy as np
import time
import requests
from monitor import ModelDriftMonitor

# --- Page Config ---
st.set_page_config(page_title="SynthMarket MLOps", layout="wide", page_icon="📈")
st.title("SynthMarket Live Ops & Drift Monitor")
st.markdown("Real-time Reinforcement Learning Trading Agent Monitoring Dashboard")

# --- State Initialization ---
if 'net_worths' not in st.session_state:
    st.session_state.net_worths = [10000.0]
    st.session_state.trade_logs = []
    st.session_state.market_data_buffer = []
    st.session_state.is_running = False

# Cache the drift monitor to avoid reloading the GAN model every render
@st.cache_resource
def get_monitor():
    return ModelDriftMonitor()
monitor = get_monitor()

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Agent Performance (Live P/L)")
    metric_cols = st.columns(2)
    pv_placeholder = metric_cols[0].empty()
    profit_placeholder = metric_cols[1].empty()
    st.markdown("<br/>", unsafe_allow_html=True)
    chart_placeholder = st.empty()
    chart_placeholder.line_chart(st.session_state.net_worths)

with col2:
    st.subheader("System Health (Model Drift)")
    status_placeholder = st.empty()
    kl_placeholder = st.empty()
    
    st.subheader("Explainable AI (Reasoning Logs)")
    log_placeholder = st.empty()

# --- Controls ---
if st.button("Start / Stop Live Simulation"):
    st.session_state.is_running = not st.session_state.is_running

# --- Live Simulation Loop ---
if st.session_state.is_running:
    # We simulate an incoming stream of market data.
    # To demonstrate the Drift Monitor, we intentionally inject a "Black Swan" anomaly halfway through.
    base_data = np.random.randn(200) * 0.2
    
    for i in range(200):
        if not st.session_state.is_running:
            break
            
        # Introduce severe market drift artificially after 100 ticks
        if i > 100:
            current_return = base_data[i] + 1.5  # Artificial shift (Drift injection!)
        else:
            current_return = base_data[i]
            
        st.session_state.market_data_buffer.append(current_return)
        if len(st.session_state.market_data_buffer) > 100:
            st.session_state.market_data_buffer.pop(0)
            
        # 1. Model Drift Monitoring (KL Divergence) - Only check after collecting enough data
        if len(st.session_state.market_data_buffer) > 30:
            drift_alert, kl_div = monitor.check_drift(st.session_state.market_data_buffer, threshold=0.4)
            
            if drift_alert:
                status_placeholder.error("🚨 DRIFT ALERT: Market behavior diverging from training data. Agent paused!")
                kl_placeholder.metric("KL Divergence", f"{kl_div:.4f} (High)", delta="Drift Detected", delta_color="inverse")
                st.warning("Auto-Pause engaged to prevent catastrophic losses.")
                st.session_state.is_running = False
                break
            else:
                status_placeholder.success("✅ HEALTHY: Market aligns with expected distribution.")
                kl_placeholder.metric("KL Divergence", f"{kl_div:.4f} (Normal)", delta="Stable")
        else:
            status_placeholder.info(f"⏳ Calibrating baseline distribution... ({len(st.session_state.market_data_buffer)}/30)")
            kl_placeholder.metric("KL Divergence", "Calculating...")
            
        # 2. XAI & Inference API Call
        try:
            # Mocking the features list
            features = list(np.random.randn(20)) + [st.session_state.net_worths[-1], 0]
            import os
            api_url = os.getenv("API_URL", "http://localhost:8000/predict")
            resp = requests.post(api_url, json={"features": features})
            if resp.status_code == 200:
                agent_action = resp.json()
            else:
                raise Exception("API error")
        except:
            # Mock fallback if FastAPI server is not running during the demo
            action_code = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            agent_action = {
                "action_name": action_map[action_code],
                "confidence": np.random.uniform(0.7, 0.99),
                "reasoning": "High Volatility (Crash incoming) detected. Liquidating assets." if action_code == 2 else ("Normal holding pattern" if action_code == 0 else "Favorable entry point identified.")
            }
            
        # 3. Process Trade and Update P/L
        current_nw = st.session_state.net_worths[-1]
        if agent_action["action_name"] == "Buy":
            new_nw = current_nw + np.random.uniform(20, 150)
        elif agent_action["action_name"] == "Sell":
            new_nw = current_nw - np.random.uniform(10, 80)
        else:
            new_nw = current_nw + np.random.uniform(-5, 5)
            
        st.session_state.net_worths.append(new_nw)
        
        # Update metrics dynamically
        pv_placeholder.metric("Portfolio Value", f"${new_nw:,.2f}", f"${new_nw - 10000:,.2f}")
        profit_placeholder.metric("Total Return", f"{((new_nw - 10000)/10000)*100:.2f}%")
        
        chart_placeholder.line_chart(st.session_state.net_worths)
        
        # 4. Update Explainability Logs
        if agent_action["action_name"] != "Hold":
            color = "green" if agent_action["action_name"] == "Buy" else "red"
            log = f"- <span style='color:{color}'>**{agent_action['action_name']}**</span> (Conf: {agent_action['confidence']:.2f}) <br/> *Reason:* {agent_action['reasoning']}"
            st.session_state.trade_logs.insert(0, log)
            if len(st.session_state.trade_logs) > 5:
                st.session_state.trade_logs.pop()
                
        with log_placeholder.container():
            for log in st.session_state.trade_logs:
                st.markdown(log, unsafe_allow_html=True)
                
        time.sleep(0.1) # Simulate real-time data streaming
