import streamlit as st
import numpy as np
import time
import requests
import plotly.graph_objects as go
from monitor import ModelDriftMonitor
import os

# --- Page Config ---
st.set_page_config(page_title="SynthMarket Pro", layout="wide", page_icon="⚡")

# --- Custom Tailwind-Inspired UI Overhaul ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Streamlit Cruft */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background Override */
    .stApp {
        background-color: #09090b !important; /* Tailwind zinc-950 */
    }

    /* Premium Tailwind-style Cards */
    .tw-card {
        background: rgba(24, 24, 27, 0.7); /* Tailwind zinc-900 with opacity */
        border: 1px solid rgba(63, 63, 70, 0.5); /* Tailwind zinc-700 */
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }

    /* Metric Values */
    .tw-metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #a1a1aa; /* Tailwind zinc-400 */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .tw-metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #fafafa; /* Tailwind zinc-50 */
    }
    .tw-metric-delta-up {
        font-size: 1rem;
        font-weight: 600;
        color: #10b981; /* Tailwind emerald-500 */
        background: rgba(16, 185, 129, 0.1);
        padding: 2px 8px;
        border-radius: 9999px;
        display: inline-block;
        margin-left: 12px;
    }
    .tw-metric-delta-down {
        font-size: 1rem;
        font-weight: 600;
        color: #ef4444; /* Tailwind red-500 */
        background: rgba(239, 68, 68, 0.1);
        padding: 2px 8px;
        border-radius: 9999px;
        display: inline-block;
        margin-left: 12px;
    }

    /* Fancy Headers */
    .tw-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f4f4f5;
        margin-bottom: 16px;
        border-bottom: 1px solid #27272a;
        padding-bottom: 12px;
    }

    /* Logs Feed */
    .tw-log-container {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 8px;
    }
    .tw-log-item {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        display: flex;
        flex-direction: column;
        transition: all 0.2s ease;
    }
    .tw-log-item:hover {
        border-color: #3f3f46;
        transform: translateX(2px);
    }
    .tw-log-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .tw-log-badge-buy {
        font-size: 0.75rem;
        font-weight: 700;
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .tw-log-badge-sell {
        font-size: 0.75rem;
        font-weight: 700;
        color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .tw-log-reason {
        font-size: 0.875rem;
        color: #a1a1aa;
        line-height: 1.4;
    }

    /* Hide default streamlit metrics to avoid clash */
    div[data-testid="stMetric"] {
        display: none !important;
    }
    
    /* Stylish primary button */
    .stButton>button {
        border: none !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #f4f4f5; font-weight: 700; font-size: 2.5rem; margin-bottom: 0;'>SynthMarket Pro ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a1a1aa; font-size: 1.1rem; margin-bottom: 2rem;'>Autonomous Quantitative Trading & Drift Observability</p>", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("<div class='tw-header'>⚙️ Control Center</div>", unsafe_allow_html=True)
    sim_speed = st.slider("Execution Latency (ms)", min_value=50, max_value=1000, value=200, step=50)
    drift_sensitivity = st.slider("Drift Threshold (KL)", min_value=0.1, max_value=1.5, value=0.4, step=0.05)
    market_scenario = st.selectbox("Market Condition", ["Random Walk (Normal)", "High Volatility", "Black Swan Crash"])
    
    st.markdown("<br><div class='tw-header'>🤖 Agent Params</div>", unsafe_allow_html=True)
    confidence_threshold = st.slider("Min Trade Confidence", min_value=0.5, max_value=0.99, value=0.7, step=0.05)

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

# --- Controls ---
button_label = "🛑 EMERGENCY STOP" if st.session_state.is_running else "▶️ INITIALIZE TRADING ALGORITHM"
if st.button(button_label, use_container_width=True):
    st.session_state.is_running = not st.session_state.is_running
    if st.session_state.is_running and st.session_state.drift_triggered:
        # Reset on restart
        st.session_state.net_worths = [10000.0]
        st.session_state.trade_logs = []
        st.session_state.market_data_buffer = []
        st.session_state.drift_triggered = False
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns([7, 3], gap="large")

with col1:
    metrics_placeholder = st.empty()
    st.markdown("<div class='tw-card'>", unsafe_allow_html=True)
    chart_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    status_placeholder = st.empty()
    log_placeholder = st.empty()

def format_metric_html(pv, diff, pct):
    color_class = "tw-metric-delta-up" if diff >= 0 else "tw-metric-delta-down"
    sign = "+" if diff >= 0 else ""
    html = f"""
    <div class="tw-card" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="tw-metric-label">Live Portfolio Value</div>
            <div class="tw-metric-value">${pv:,.2f} <span class="{color_class}">{sign}${abs(diff):,.2f} ({sign}{pct:.2f}%)</span></div>
        </div>
    </div>
    """
    return html

def format_status_html(is_alert, kl_div):
    if kl_div is None:
        title = "⏳ Calibrating..."
        color = "#3b82f6"
        kl_text = "N/A"
    elif is_alert:
        title = "🚨 ANOMALY DETECTED"
        color = "#ef4444"
        kl_text = f"{kl_div:.4f}"
    else:
        title = "✅ SYSTEM OPTIMAL"
        color = "#10b981"
        kl_text = f"{kl_div:.4f}"
        
    html = f"""
    <div class="tw-card">
        <div class="tw-metric-label">System Health (Drift Monitor)</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {color}; margin-bottom: 10px;">{title}</div>
        <div class="tw-metric-label" style="margin-bottom: 0;">KL Divergence: <span style="color: #f4f4f5; font-weight: 600;">{kl_text}</span></div>
    </div>
    """
    return html

def format_logs_html(logs):
    html = "<div class='tw-card'><div class='tw-header'>⚡ Inference Feed</div><div class='tw-log-container'>"
    if not logs:
        html += "<div class='tw-log-reason'>Awaiting signals...</div>"
    for log in logs:
        badge_class = "tw-log-badge-buy" if log['action'] == "Buy" else "tw-log-badge-sell"
        html += f"""
        <div class="tw-log-item">
            <div class="tw-log-header">
                <span class="{badge_class}">{log['action']}</span>
                <span style="font-size: 0.75rem; color: #71717a;">Conf: {log['conf']:.0f}%</span>
            </div>
            <div class="tw-log-reason">{log['reason']}</div>
        </div>
        """
    html += "</div></div>"
    return html

# Initial Chart Render
def render_ui(kl_div=None, is_alert=False):
    y_data = st.session_state.net_worths
    x_data = list(range(len(y_data)))
    pv = y_data[-1]
    diff = pv - 10000.0
    pct = (diff / 10000.0) * 100
    
    # 1. Update Metrics
    metrics_placeholder.markdown(format_metric_html(pv, diff, pct), unsafe_allow_html=True)
    
    # 2. Update Status
    status_placeholder.markdown(format_status_html(is_alert, kl_div), unsafe_allow_html=True)
    
    # 3. Update Logs
    log_placeholder.markdown(format_logs_html(st.session_state.trade_logs), unsafe_allow_html=True)
    
    # 4. Render Chart
    fig = go.Figure()
    line_color = "#10b981" if pv >= 10000 else "#ef4444"
    fill_color = "rgba(16, 185, 129, 0.1)" if pv >= 10000 else "rgba(239, 68, 68, 0.1)"
    
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='lines',
        line=dict(color=line_color, width=3, shape='spline'),
        fill='tozeroy',
        fillcolor=fill_color,
        name='Portfolio Value',
        hovertemplate='<b>Tick:</b> %{x}<br><b>Value:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    if st.session_state.drift_triggered:
        fig.add_trace(go.Scatter(
            x=[x_data[-1]], y=[y_data[-1]],
            mode='markers',
            marker=dict(color='#ef4444', size=14, symbol='x-open-dot', line=dict(width=3)),
            name='Drift Alert'
        ))
        
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(63,63,70,0.3)', tickprefix="$"),
        hovermode='x unified',
        showlegend=False,
        height=400
    )
    min_val = min(9800, min(y_data) - 100)
    max_val = max(10200, max(y_data) + 100)
    fig.update_yaxes(range=[min_val, max_val])
    chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if not st.session_state.is_running:
    render_ui()
    
# --- Live Simulation Loop ---
if st.session_state.is_running and not st.session_state.drift_triggered:
    if market_scenario == "Random Walk (Normal)":
        mu, sigma = 0.0001, 0.01
    elif market_scenario == "High Volatility":
        mu, sigma = 0.0001, 0.03
    else: 
        mu, sigma = 0.0001, 0.01
        
    for i in range(200):
        if not st.session_state.is_running:
            break
            
        dW = np.random.normal(0, 1)
        if market_scenario == "Black Swan Crash" and i > 100:
            current_return = np.random.normal(-0.02, 0.05)
        else:
            current_return = mu + sigma * dW
            
        st.session_state.market_data_buffer.append(current_return)
        if len(st.session_state.market_data_buffer) > 100:
            st.session_state.market_data_buffer.pop(0)
            
        # Model Drift Monitoring
        kl_val = None
        drift_alert = False
        if len(st.session_state.market_data_buffer) > 30:
            drift_alert, kl_val = monitor.check_drift(st.session_state.market_data_buffer, threshold=drift_sensitivity)
            if drift_alert:
                st.session_state.is_running = False
                st.session_state.drift_triggered = True
                
        if st.session_state.drift_triggered:
            render_ui(kl_div=kl_val, is_alert=True)
            break
            
        # XAI & Inference API Call
        try:
            features = list(np.random.randn(20)) + [st.session_state.net_worths[-1], 0]
            api_url = os.getenv("API_URL", "http://localhost:8000/predict")
            resp = requests.post(api_url, json={"features": features}, timeout=2)
            if resp.status_code == 200:
                agent_action = resp.json()
            else:
                raise Exception("API error")
        except:
            action_code = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            agent_action = {
                "action_name": action_map[action_code],
                "confidence": float(np.random.uniform(0.7, 0.99)),
                "reasoning": "High Volatility detected. Liquidating assets." if action_code == 2 else ("Normal holding pattern" if action_code == 0 else "Favorable entry point identified.")
            }
            
        # Process Trade and Update P/L
        current_nw = st.session_state.net_worths[-1]
        if agent_action["action_name"] == "Buy":
            new_nw = current_nw * (1.0 + current_return) + np.random.uniform(5, 20)
            st.toast(f"Trade Executed: BUY", icon="🟢")
        elif agent_action["action_name"] == "Sell":
            new_nw = current_nw * (1.0 + current_return) - np.random.uniform(2, 10)
            st.toast(f"Trade Executed: SELL", icon="🔴")
        else:
            new_nw = current_nw * (1.0 + current_return)
            
        st.session_state.net_worths.append(new_nw)
        
        # Update Logs
        if agent_action["action_name"] != "Hold" and agent_action["confidence"] >= confidence_threshold:
            log_entry = {
                "action": agent_action['action_name'],
                "conf": agent_action['confidence'] * 100,
                "reason": agent_action['reasoning']
            }
            st.session_state.trade_logs.insert(0, log_entry)
            if len(st.session_state.trade_logs) > 6:
                st.session_state.trade_logs.pop()
                
        # Render UI every 2 ticks
        if i % 2 == 0:
            render_ui(kl_div=kl_val, is_alert=drift_alert)
        
        time.sleep(sim_speed / 1000.0)
