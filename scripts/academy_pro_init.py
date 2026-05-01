import sys
import os
from datetime import datetime

# Ajouter le chemin src au PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), 'src'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from quantum.infrastructure.db.models import Base, Course, Lesson, Quiz, Question, Option
from quantum.shared.config.settings import config

def init_academy_pro():
    print("🚀 Initialisation de l'Académie Pro...")
    
    engine = create_engine(config.database.DATABASE_URL_SYNC)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Nettoyage optionnel (à décommenter si on veut reset)
    # session.query(Option).delete()
    # session.query(Question).delete()
    # session.query(Quiz).delete()
    # session.query(Lesson).delete()
    # session.query(Course).delete()
    # session.commit()

    # --- MODULE 1: BASES DU TRADING ---
    course1 = Course(
        title="Bases du Trading Institutionnel",
        description="Apprenez les fondamentaux des marchés financiers et les types de trading.",
        level="Débutant",
        order=1
    )
    session.add(course1)
    session.flush()

    lesson1_1 = Lesson(
        course_id=course1.id,
        title="1. C'est quoi le trading ?",
        content="""# 🧠 C'est quoi le trading ?

Le trading consiste à acheter et vendre des actifs financiers pour faire un profit.

### 👉 Les marchés principaux :
*   📈 **Forex** (devises : EUR/USD…)
*   📊 **Actions** (Apple, Tesla…)
*   🪙 **Crypto** (Bitcoin…)
*   📉 **Indices** (S&P 500, NASDAQ)
*   🛢️ **Matières premières** (or, pétrole…)
""",
        order=1,
        duration="5 min"
    )
    
    lesson1_2 = Lesson(
        course_id=course1.id,
        title="2. Les types de trading",
        content="""# 📊 Les types de trading

### 🔹 Scalping
*   Trades très rapides (secondes/minutes)
*   Beaucoup de trades par jour
*   Très stressant ⚠️

### 🔹 Day Trading
*   Tu ouvres et fermes dans la même journée
*   Pas de position la nuit

### 🔹 Swing Trading
*   Trades sur plusieurs jours/semaines
*   Moins stressant (idéal débutant 👍)

### 🔹 Investing
*   Long terme (mois/années)
*   Moins actif
""",
        order=2,
        duration="10 min"
    )

    lesson1_3 = Lesson(
        course_id=course1.id,
        title="3. Les bases essentielles",
        content="""# 📚 Les bases essentielles à connaître

### 🔑 Termes importants
*   **Lot** : taille de position
*   **Pip** : plus petit mouvement de prix
*   **Spread** : différence achat/vente
*   **Levier (leverage)** : multiplier ton capital (⚠️ dangereux)
*   **Stop Loss (SL)** : limite tes pertes
*   **Take Profit (TP)** : sécurise tes gains
*   **Drawdown** : perte totale temporaire
""",
        order=3,
        duration="10 min"
    )

    session.add_all([lesson1_1, lesson1_2, lesson1_3])

    # Quiz Module 1
    quiz1 = Quiz(course_id=course1.id, title="Test de Niveau 1: Les Bases", description="Validez vos connaissances sur les fondamentaux.")
    session.add(quiz1)
    session.flush()

    q1_data = [
        ("Le marché le plus liquide au monde est :", ["Crypto", "Forex", "Actions", "Matières premières"], 1),
        ("Le spread correspond à :", ["Une taxe", "La différence entre achat et vente", "Un indicateur", "Un profit"], 1),
        ("Une position SELL signifie :", ["Acheter", "Attendre", "Vendre", "Bloquer"], 2),
        ("Le Stop Loss sert à :", ["Augmenter les gains", "Limiter les pertes", "Ouvrir une position", "Suivre le marché"], 1),
        ("Le marché crypto est ouvert :", ["24h/24 toute la semaine", "Seulement la journée", "Le week-end uniquement"], 0),
    ]

    for q_text, opts, correct_idx in q1_data:
        q = Question(quiz_id=quiz1.id, text=q_text)
        session.add(q)
        session.flush()
        for i, opt_text in enumerate(opts):
            session.add(Option(question_id=q.id, text=opt_text, is_correct=(i == correct_idx)))

    # --- MODULE 2: ANALYSE TECHNIQUE ET STRATÉGIE ---
    course2 = Course(
        title="Analyse Technique et Psychologie",
        description="Maîtrisez les graphiques, les indicateurs et la psychologie du trader.",
        level="Intermédiaire",
        order=2
    )
    session.add(course2)
    session.flush()

    lesson2_1 = Lesson(
        course_id=course2.id,
        title="4. Analyse technique",
        content="""# 📉 Analyse technique (TRÈS IMPORTANT)

### 🔹 Support & Résistance
*   **Support** = zone où le prix rebondit vers le haut
*   **Résistance** = zone où le prix redescend

### 🔹 Tendances
*   📈 **Bullish** = marché monte
*   📉 **Bearish** = marché descend

### 🔹 Indicateurs populaires
*   **RSI** (surachat / survente)
*   **MACD** (momentum)
*   **Moyennes mobiles**
""",
        order=1,
        duration="15 min"
    )

    lesson2_2 = Lesson(
        course_id=course2.id,
        title="5. Gestion du Risque",
        content="""# 💰 Gestion du risque (LE PLUS IMPORTANT)

Sans ça → tu perds.

### Règles de base :
*   ❌ Ne jamais risquer plus de 1-2% par trade
*   ✔️ Toujours mettre un Stop Loss
*   ✔️ Ratio risque/gain minimum 1:2
""",
        order=2,
        duration="10 min"
    )

    lesson2_3 = Lesson(
        course_id=course2.id,
        title="6. Psychologie du Trader",
        content="""# 🧠 Psychologie du trader

Le vrai combat est ici :
*   **Émotions** (peur, greed)
*   **Discipline**
*   **Patience**

### 👉 Erreurs fréquentes :
*   Trader sans plan
*   Overtrading
*   Vouloir se refaire après une perte
""",
        order=3,
        duration="10 min"
    )

    session.add_all([lesson2_1, lesson2_2, lesson2_3])

    # Quiz Module 2
    quiz2 = Quiz(course_id=course2.id, title="Test de Niveau 2: Technique & Risque", description="Prouvez votre maîtrise de l'analyse technique.")
    session.add(quiz2)
    session.flush()

    q2_data = [
        ("Un marché bullish est :", ["En baisse", "Stable", "En hausse", "Fermé"], 2),
        ("Le RSI mesure :", ["Le volume", "La vitesse du prix", "Surachat / survente", "Le spread"], 2),
        ("Un ratio risque/gain de 1:2 signifie :", ["Risquer 2 pour gagner 1", "Risquer 1 pour gagner 2", "Aucun risque"], 1),
        ("Le drawdown représente :", ["Un gain", "Une perte temporaire", "Une stratégie", "Une taxe"], 1),
        ("Tu risques 2% de 1000$. Combien risques-tu ?", ["10$", "20$", "50$", "100$"], 1),
    ]

    for q_text, opts, correct_idx in q2_data:
        q = Question(quiz_id=quiz2.id, text=q_text)
        session.add(q)
        session.flush()
        for i, opt_text in enumerate(opts):
            session.add(Option(question_id=q.id, text=opt_text, is_correct=(i == correct_idx)))

    # --- MODULE 3: TEST PROP FIRM ---
    course3 = Course(
        title="Simulation Réelle : Test Prop Firm",
        description="Mettez-vous dans la peau d'un trader professionnel financé.",
        level="Avancé",
        order=3
    )
    session.add(course3)
    session.flush()

    lesson3_1 = Lesson(
        course_id=course3.id,
        title="Règles du Test Prop Firm",
        content="""# 🧠 📊 TEST PROP FIRM (SIMULATION RÉELLE)

### ⚠️ RÈGLES DU TEST
*   **Capital** : 10 000$
*   **Perte max par jour** : 5% (500$)
*   **Perte max totale** : 10% (1000$)
*   **Objectif** : +10% (1000$)
*   **Risque par trade conseillé** : 1%
""",
        order=1,
        duration="5 min"
    )
    session.add(lesson3_1)

    quiz3 = Quiz(course_id=course3.id, title="Examen Final: Prop Firm Challenge", description="Réussirez-vous à obtenir votre financement ?")
    session.add(quiz3)
    session.flush()

    q3_data = [
        ("Tu es à -4.5% de perte sur la journée. Prends-tu un nouveau trade ?", ["Oui", "Non (tu es presque à la limite)"], 1),
        ("Tu as atteint +10% en 2 jours. Que fais-tu ?", ["Tu continues pour gagner plus", "Tu stoppes (objectif atteint)"], 1),
        ("Tu fais 5 trades en 1 heure. Quel est le problème ?", ["Aucun, c'est normal", "Overtrading"], 1),
        ("Ratio 1:3 avec 50% de réussite est-il rentable ?", ["Oui", "Non"], 0),
        ("Quelle qualité fait durer un trader ?", ["L'intelligence", "La chance", "La discipline"], 2),
    ]

    for q_text, opts, correct_idx in q3_data:
        q = Question(quiz_id=quiz3.id, text=q_text)
        session.add(q)
        session.flush()
        for i, opt_text in enumerate(opts):
            session.add(Option(question_id=q.id, text=opt_text, is_correct=(i == correct_idx)))

    session.commit()
    print("✅ Académie Pro initialisée avec succès !")

if __name__ == "__main__":
    init_academy_pro()
