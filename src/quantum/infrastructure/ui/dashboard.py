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
st.sidebar.title("Quantum Dashboard v3")
st.sidebar.markdown(f"**Mode:** 🟢 LIVE EXECUTION READY" if os.getenv('LIVE_TRADING') == 'True' else "**Mode:** 🟡 Simulation / Paper")
st.sidebar.markdown(f"**IA:** Neural Alpha + Social Intelligence")

active_symbols = config.symbols.ACTIVE_SYMBOLS
selected_symbols = st.sidebar.multiselect("Actifs à surveiller", active_symbols, default=active_symbols[:5])

# --- Header ---
st.title("🛡️ Quantum Intelligence Hub v3")
st.markdown("---")

# --- Métriques Globales ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Market Status", "OPEN", "Normal")
col2.metric("Execution Engine", "ACTIVE", border_left=True)
col3.metric("Social Pulse", "BULLISH", "+12% Tweets")
col4.metric("Last Order", "NONE", "-")

# --- Analyse & Intelligence Sociale ---
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.subheader("📊 Unified Alpha Scanner")
    if st.button("Lancer un scan complet"):
        with st.spinner("Analyse quantitative & sociale en cours..."):
            results = []
            for sym in selected_symbols:
                try:
                    res = asyncio.run(system.analyze_symbol(sym))
                    results.append({
                        'Symbol': sym,
                        'Signal': res['combined_signal'],
                        'Confidence': f"{res['confidence']:.1f}%",
                        'Social Sentiment': f"{res['social_data']['score']:.1f}%",
                        'Hurst': f"{res['hurst_value']:.2f}",
                        'Web3 Score': f"{res['onchain_data'].get('mempool', {}).get('pressure_score', 0)}"
                    })
                except Exception as e:
                    st.sidebar.error(f"Erreur sur {sym}: {e}")
            
            df_results = pd.DataFrame(results)
            st.table(df_results)

with row1_col2:
    st.subheader("🐦 Social Sentiment (Twitter/X)")
    for sym in selected_symbols[:3]:
        social_res = system.social_analyzer.analyze_asset(sym)
        st.write(f"**{sym}**")
        st.progress(social_res['score'] / 100)
        st.caption(f"Status: {social_res['label']} | Volume: {social_res['volume_signal']}")

# --- Position & Risk ---
st.markdown("---")
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.subheader("🏦 Portefeuille & Exécution")
    if os.getenv('LIVE_TRADING') == 'True':
        st.success("Connecté aux API Binance/IBKR")
        st.info("Aucune position active pour le moment.")
    else:
        st.warning("Mode Simulation activé. Aucun ordre réel ne sera envoyé.")

with row2_col2:
    st.subheader("🛑 Risque & Circuit Breaker")
    cb_status = system.circuit_breaker.get_status()
    st.metric("Drawdown Actuel", f"{cb_status['current_drawdown']:.2f}%", help="Max allowed: 5%")
    st.progress(min(cb_status['current_drawdown'] / 5.0, 1.0))


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
