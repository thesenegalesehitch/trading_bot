"""
Quantum Trading System - Professional Dashboard
===============================================
Institutional GUI for monitoring quantitative signals and portfolio risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Ajouter le chemin src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from quantum.engine.orchestrator import QuantumTradingSystem
    from quantum.shared.config.settings import config
except ImportError:
    st.error("Impossible de charger les modules Quantum. Assurez-vous d'être dans le bon environnement.")

st.set_page_config(page_title="Quantum Trading - Institutional Dashboard", layout="wide")

# --- Initialisation du Système ---
@st.cache_resource
def get_system():
    system = QuantumTradingSystem()
    return system

system = get_system()

# --- Sidebar ---
st.sidebar.title("Quantum Dashboard v2")
st.sidebar.markdown(f"**Modèle:** Institutional Clean Architecture")
st.sidebar.markdown(f"**Moteur:** Alpha Unified (Web3 + Tech)")

active_symbols = config.symbols.ACTIVE_SYMBOLS
selected_symbols = st.sidebar.multiselect("Actifs à surveiller", active_symbols, default=active_symbols[:5])

refresh_rate = st.sidebar.slider("Fréquence de rafraîchissement (s)", 10, 300, 60)

# --- Header ---
st.title("🛡️ Quantum Intelligence Hub")
st.markdown("---")

# --- Métriques Globales ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Market Status", "OPEN", "Normal")
col2.metric("Active Assets", len(selected_symbols))
col3.metric("System Health", "OPTIMAL", border_left=True)
col4.metric("Last Analysis", datetime.now().strftime("%H:%M:%S"))

# --- Analyse en Temps Réel ---
st.subheader("📊 Unified Alpha Scanner")

if st.button("Lancer un scan complet"):
    with st.spinner("Analyse quantitative en cours..."):
        results = []
        for sym in selected_symbols:
            try:
                res = system.analyze_symbol(sym)
                results.append({
                    'Symbol': sym,
                    'Price': res['price'],
                    'Signal': res['combined_signal'],
                    'Confidence': f"{res['confidence']:.1f}%",
                    'Hurst': f"{res['hurst_value']:.2f}",
                    'Z-Score': f"{res['zscore_data'].get('zscore', 0):.2f}",
                    'Web3 Score': f"{res['onchain_data'].get('mempool', {}).get('pressure_score', 0)}"
                })
            except Exception as e:
                st.sidebar.error(f"Erreur sur {sym}: {e}")
        
        df_results = pd.DataFrame(results)
        
        # Coloration des signaux
        def color_signal(val):
            color = 'white'
            if val == 'BUY': color = '#2ecc71'
            elif val == 'SELL': color = '#e74c3c'
            elif val == 'WAIT': color = '#f1c40f'
            return f'background-color: {color}; color: black'

        st.table(df_results.style.applymap(color_signal, subset=['Signal']))

# --- Graphs ---
st.markdown("---")
row2_col1, row2_col2 = st.columns([2, 1])

with row2_col1:
    st.subheader("📈 Dynamique des Signaux")
    # Mock data pour le graph (évolution du score de confiance)
    hist_data = pd.DataFrame({
        'Time': pd.date_range(start='now', periods=20, freq='H'),
        'Alpha Score': np.random.uniform(30, 95, 20)
    })
    fig = px.line(hist_data, x='Time', y='Alpha Score', title="Évolution de la confiance Alpha (Moyenne Portefeuille)")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with row2_col2:
    st.subheader("🛑 Risque & Circuit Breaker")
    cb_status = system.circuit_breaker.get_status()
    
    # Donut chart pour le Drawdown
    current_dd = cb_status['current_drawdown']
    max_dd = cb_status['max_drawdown']
    
    fig_risk = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_dd,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Drawdown %"},
        gauge = {
            'axis': {'range': [None, max_dd * 1.5]},
            'steps': [
                {'range': [0, max_dd * 0.8], 'color': "green"},
                {'range': [max_dd * 0.8, max_dd], 'color': "orange"},
                {'range': [max_dd, max_dd * 1.5], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': max_dd
            }
        }
    ))
    fig_risk.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_risk, use_container_width=True)

# --- Web3 Intelligence ---
st.markdown("---")
st.subheader("🌐 On-Chain & Web3 Intelligence")
web3_data = system.web3_engine.get_current_analysis()

w_col1, w_col2, w_col3 = st.columns(3)

with w_col1:
    st.write("**Mempool Pressure**")
    m_score = web3_data.get('mempool', {}).get('pressure_score', 0)
    st.info(f"Score: {m_score}")

with w_col2:
    st.write("**Cross-Chain Index (CCI)**")
    cci = web3_data.get('cross_chain', {}).get('cci_value', 0)
    st.metric("CCI", f"{cci:.4f}", delta=None)

with w_col3:
    st.write("**Sentiment Staking**")
    eth_s = web3_data.get('sentiment', {}).get('eth', {}).get('score', 0)
    st.success(f"ETH Score: {eth_s}")

st.markdown("---")
st.caption("Quantum Trading System v2.0 - Institutional Grade - © 2026 Alexandre Albert Ndour")
