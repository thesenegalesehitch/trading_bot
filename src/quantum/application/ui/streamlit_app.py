"""
Trade Advisor & Coach - Streamlit UI
Phase 5: Polish + Launch - Trade Advisor & Coach v2

Interface Streamlit pour le coach de trading AI.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# Import des modules
from quantum.domain.coach.validator import TradeValidator
from quantum.domain.coach.explainer import LLMExplainer, SignalDirection
from quantum.domain.coach.history import TradeHistory, Trade, TradeOutcome
from quantum.domain.innovations.whatif_simulator import WhatIfSimulator, SimulationType
from quantum.domain.innovations.postmortem import AutoPostMortem
from quantum.domain.innovations.mistake_predictor import MistakePredictor
from quantum.domain.innovations.confusion_resolver import ConfusionResolver


# Configuration de la page
st.set_page_config(
    page_title="Trade Advisor & Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main function for the Streamlit app."""
    
    # Header
    st.markdown('<p class="main-header">🎯 Trade Advisor & Coach</p>', unsafe_allow_html=True)
    st.markdown("### Votre Copilote de Trading AI")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Menu principal
    menu = st.sidebar.selectbox(
        "Menu",
        [
            "🏠 Accueil",
            "🔮 Confusion Resolver",
            "✅ Validate Trade",
            "📊 Trade History",
            "🔬 What If Simulator",
            "📝 Post-Mortem",
            "⚠️ Mistake Predictor"
        ]
    )
    
    if menu == "🏠 Accueil":
        home_page()
    elif menu == "🔮 Confusion Resolver":
        confusion_resolver_page()
    elif menu == "✅ Validate Trade":
        validate_trade_page()
    elif menu == "📊 Trade History":
        trade_history_page()
    elif menu == "🔬 What If Simulator":
        whatif_page()
    elif menu == "📝 Post-Mortem":
        postmortem_page()
    elif menu == "⚠️ Mistake Predictor":
        mistake_predictor_page()


def home_page():
    """Page d'accueil."""
    st.markdown("## Bienvenue sur Trade Advisor & Coach")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔮 Confusion Resolver")
        st.markdown("Resolve les contradictions d'indicateurs")
        st.info("FLAGSHP - Le resolveur de confusion")
    
    with col2:
        st.markdown("### ✅ Validate Trade")
        st.markdown("Valide tes trades avant d'entrer")
        st.info("Feedback structuré instantané")
    
    with col3:
        st.markdown("### 📊 Trade History")
        st.markdown("Suis tes performances")
        st.info("Analytics et statistiques")


def confusion_resolver_page():
    """Page du Confusion Resolver."""
    st.markdown("## 🔮 Confusion Resolver")
    st.markdown("### Le FLAGSHP - Résout les contradictions d'indicateurs")
    
    # Input
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbole", value="EURUSD")
    with col2:
        timeframe = st.selectbox("Timeframe", ["15M", "1H", "4H", "1D"])
    
    if st.button("Analyser", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                resolver = ConfusionResolver()
                result = resolver.resolve(symbol, timeframe)
                
                # Afficher le signal
                if result.final_signal == "BUY":
                    st.success(f"🟢 SIGNAL: BUY ({result.confidence:.0f}% confiance)")
                elif result.final_signal == "SELL":
                    st.error(f"🔴 SIGNAL: SELL ({result.confidence:.0f}% confiance)")
                else:
                    st.warning(f"⚪ SIGNAL: NEUTRAL ({result.confidence:.0f}% confiance)")
                
                st.markdown(f"**Régime de marché:** {result.regime}")
                
                # Pondération
                st.markdown("### Pondération des indicateurs:")
                for name, weight in result.weighting.items():
                    st.write(f"- {name}: {weight}")
                
                # Explication
                st.markdown("### Explication:")
                st.markdown(result.explanation)
                
                # Verdict
                st.markdown("---")
                st.markdown(f"### 📋 VERDICT: {result.verdict}")
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")


def validate_trade_page():
    """Page de validation de trade."""
    st.markdown("## ✅ Validate Trade")
    st.markdown("Valide ton trade avant d'entrer")
    
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbole", value="EURUSD")
        direction = st.selectbox("Direction", ["BUY", "SELL"])
    with col2:
        entry_price = st.number_input("Prix d'entrée", value=1.0850, format="%.5f")
        stop_loss = st.number_input("Stop Loss", value=1.0820, format="%.5f")
    with col3:
        take_profit = st.number_input("Take Profit", value=1.0910, format="%.5f")
        account_balance = st.number_input("Solde du compte", value=10000.0)
    
    if st.button("Valider le trade", type="primary"):
        with st.spinner("Validation en cours..."):
            try:
                validator = TradeValidator()
                result = validator.validate_trade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    account_balance=account_balance
                )
                
                # Score
                st.metric("Score de validation", f"{result.score}/100")
                
                if result.is_valid:
                    st.success("✅ Trade valide")
                else:
                    st.error("❌ Trade invalide")
                
                st.markdown(f"**Résumé:** {result.summary}")
                
                # Issues
                if result.issues:
                    st.markdown("### Problèmes détectés:")
                    for issue in result.issues:
                        if issue.severity.value == "error":
                            st.error(f"**{issue.category}**: {issue.message}")
                        elif issue.severity.value == "warning":
                            st.warning(f"**{issue.category}**: {issue.message}")
                        else:
                            st.info(f"**{issue.category}**: {issue.message}")
                        st.write(f"→ {issue.recommendation}")
                
                # Improvements
                if result.improvements:
                    st.markdown("### Améliorations suggérées:")
                    for imp in result.improvements:
                        st.write(f"- {imp}")
                        
            except Exception as e:
                st.error(f"Erreur: {str(e)}")


def trade_history_page():
    """Page de l'historique des trades."""
    st.markdown("## 📊 Trade History")
    
    # Charger l'historique
    history = TradeHistory()
    
    # Statistiques
    stats = history.get_statistics(30)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", stats['total_trades'])
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    with col3:
        st.metric("P&L Total", f"{stats['total_pnl']:.2f}%")
    with col4:
        st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    
    # Analyse par setup
    st.markdown("### Performance par Setup")
    setup_stats = history.analyze_by_setup()
    if setup_stats:
        df = pd.DataFrame(setup_stats).T
        st.dataframe(df)
    else:
        st.info("Aucun trade dans l'historique")
    
    # Trades récents
    st.markdown("### Trades récents")
    recent = history.get_recent_trades(10)
    if recent:
        for trade in recent:
            emoji = "🟢" if trade.outcome == TradeOutcome.WIN else "🔴" if trade.outcome == TradeOutcome.LOSS else "⚪"
            st.write(f"{emoji} {trade.symbol} {trade.direction} - {trade.pnl_pips:.1f} pips - {trade.setup_name}")
    else:
        st.info("Aucun trade récent")


def whatif_page():
    """Page du What If Simulator."""
    st.markdown("## 🔬 What If Simulator")
    st.markdown("Simule ce qui se serait passé si...")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbole", value="EURUSD")
        direction = st.selectbox("Direction", ["BUY", "SELL"])
    with col2:
        entry_price = st.number_input("Prix d'entrée", value=1.0850, format="%.5f")
        entry_date = st.date_input("Date d'entrée", value=datetime.now() - timedelta(days=7))
    
    col3, col4 = st.columns(2)
    with col3:
        stop_loss = st.number_input("Stop Loss", value=1.0820, format="%.5f")
    with col4:
        take_profit = st.number_input("Take Profit", value=1.0910, format="%.5f")
    
    if st.button("Simuler", type="primary"):
        with st.spinner("Simulation en cours..."):
            try:
                simulator = WhatIfSimulator()
                result = simulator.simulate(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    entry_date=datetime.combine(entry_date, datetime.min.time()),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                st.markdown("### Résultats")
                st.write(f"**Entrée:** {result.entry_price}")
                st.write(f"**Sortie:** {result.exit_price}")
                st.write(f"**Raison de sortie:** {result.exit_reason}")
                st.write(f"**P&L:** {result.pnl_pips} pips ({result.pnl_percent}%)")
                st.write(f"**Profit max:** {result.max_profit_pips} pips")
                st.write(f"**Perte max:** {result.max_loss_pips} pips")
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")


def postmortem_page():
    """Page du Post-Mortem."""
    st.markdown("## 📝 Auto Post-Mortem")
    st.markdown("Analyse automatique de tes trades")
    
    # Input du trade
    st.markdown("### Entrez les détails du trade")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbole", value="EURUSD")
        direction = st.selectbox("Direction", ["BUY", "SELL"])
    with col2:
        entry_price = st.number_input("Prix d'entrée", value=1.0850, format="%.5f")
        exit_price = st.number_input("Prix de sortie", value=1.0820, format="%.5f")
    
    col3, col4 = st.columns(2)
    with col3:
        stop_loss = st.number_input("Stop Loss", value=1.0820, format="%.5f")
    with col4:
        take_profit = st.number_input("Take Profit", value=1.0910, format="%.5f")
    
    if st.button("Analyser", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                # Créer un trade fictif
                trade = Trade(
                    id="temp",
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=datetime.now() - timedelta(days=1),
                    exit_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    quantity=1.0,
                    outcome=TradeOutcome.WIN if exit_price > entry_price else TradeOutcome.LOSS,
                    pnl=(exit_price - entry_price) / entry_price * 100,
                    pnl_pips=(exit_price - entry_price) * 10000,
                    reasoning="Analyse",
                    notes="",
                    setup_name="Manual",
                    timeframe="1H",
                    validation_score=50,
                    tags=[],
                    metadata={}
                )
                
                analyzer = AutoPostMortem()
                result = analyzer.analyze(trade)
                
                st.markdown(result.report)
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")


def mistake_predictor_page():
    """Page du Mistake Predictor."""
    st.markdown("## ⚠️ Mistake Predictor")
    st.markdown("Prédit quand tu vas faire une erreur")
    
    # Charger l'historique
    history = TradeHistory()
    
    if st.button("Analyser mon comportement", type="primary"):
        with st.spinner("Analyse en cours..."):
            try:
                predictor = MistakePredictor(history)
                prediction = predictor.predict_mistake()
                
                risk_score = predictor.get_risk_score()
                
                st.metric("Score de risque", f"{risk_score}/100")
                
                if prediction:
                    if prediction.probability >= 70:
                        st.error(f"⚠️ {prediction.alert_message}")
                    elif prediction.probability >= 50:
                        st.warning(f"⚠️ {prediction.alert_message}")
                    else:
                        st.info(f"ℹ️ {prediction.alert_message}")
                    
                    st.write(f"**Pattern:** {prediction.pattern.value}")
                    st.write(f"**Probabilité:** {prediction.probability:.0f}%")
                    st.write(f"**Recommandation:** {prediction.recommendation}")
                    st.write(f"**Cooldown:** {prediction.cooldown_minutes} minutes")
                else:
                    st.success("✅ Aucun pattern détecté - Tu es prêt à trader!")
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Stats des patterns
    st.markdown("### Statistiques des patterns")
    try:
        history_temp = TradeHistory()
        predictor = MistakePredictor(history_temp)
        pattern_stats = predictor.get_pattern_statistics()
        
        st.write(pattern_stats)
        
    except Exception:
        st.info("Pas assez de données pour les statistiques")


if __name__ == "__main__":
    main()
