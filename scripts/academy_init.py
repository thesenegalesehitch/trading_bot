import asyncio
import sys
import os

# Ajouter src au PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "src"))

from quantum.infrastructure.db.session import async_engine, async_sessionmaker
from quantum.infrastructure.db.models import Course, Lesson, Quiz, Question, Option

async def populate_academy():
    print("Populating Academy with real content...")
    AsyncSessionLocal = async_sessionmaker(bind=async_engine)
    
    async with AsyncSessionLocal() as session:
        # Module 1: Psychologie
        m1 = Course(
            title="Indépendance et Discipline",
            description="Maîtrisez l'aspect psychologique crucial pour survivre sur les marchés institutionnels.",
            level="Beginner",
            order=1,
            image_url="https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?auto=format&fit=crop&q=80&w=400"
        )
        session.add(m1)
        await session.flush()
        
        l1 = Lesson(
            course_id=m1.id,
            title="La Rigueur Institutionnelle",
            content="""# La Rigueur Institutionnelle
            
Le trading n'est pas un jeu, c'est une affaire de probabilités et de discipline. 
Les institutions ne 'devinent' pas le marché, elles exécutent des algorithmes basés sur la liquidité.

## Points Clés:
1. **La patience est une compétence** : Attendre le setup parfait.
2. **Détachement émotionnel** : Le résultat d'un trade ne définit pas votre valeur.
3. **Journal de trading** : Si vous ne mesurez pas, vous ne pouvez pas progresser.
""",
            order=1,
            duration="15 min"
        )
        session.add(l1)
        
        q1 = Quiz(course_id=m1.id, title="Quizz Psychologie", description="Vérifiez votre état d'esprit.")
        session.add(q1)
        await session.flush()
        
        ques1 = Question(quiz_id=q1.id, text="Quelle est la règle n°1 du trader institutionnel ?", explanation="La discipline surpasse l'intelligence sur les marchés.")
        session.add(ques1)
        await session.flush()
        
        session.add(Option(question_id=ques1.id, text="Avoir raison tout le temps", is_correct=False))
        session.add(Option(question_id=ques1.id, text="Suivre son plan avec discipline", is_correct=True))
        session.add(Option(question_id=ques1.id, text="Gagner 100% de ses trades", is_correct=False))

        # Module 2: Risk Management
        m2 = Course(
            title="Gestion du Risque & VaR",
            description="Le secret des pros : comment ne jamais 'burn' son compte.",
            level="Intermediate",
            order=2
        )
        session.add(m2)
        await session.flush()
        
        l2 = Lesson(
            course_id=m2.id,
            title="Calculer son risque par trade",
            content="""# Calculer son risque par trade
            
Le risque par trade ne devrait jamais excéder 1% à 2% de votre capital total.

## La Formule Magique:
`Taille de Position = Risque ($) / Distance Stop Loss (pips)`

Utilisez notre outil de calcul de risque pour automatiser ce processus.
""",
            order=1,
            duration="20 min"
        )
        session.add(l2)

        # Module 4: SMC/ICT
        m4 = Course(
            title="Smart Money Concepts (SMC)",
            description="Détecter les traces des banques via les FVG et les Order Blocks.",
            level="Advanced",
            order=4
        )
        session.add(m4)
        await session.flush()
        
        l4 = Lesson(
            course_id=m4.id,
            title="Fair Value Gaps (FVG)",
            content="""# Fair Value Gaps (FVG)
            
Un FVG est une inefficacité de prix créée par un mouvement violent. 
Le marché a tendance à revenir combler ces zones.

![FVG Example](https://images.unsplash.com/photo-1611974717482-480061d496a7?auto=format&fit=crop&q=80&w=600)

## Comment les repérer ?
Une bougie à corps long qui laisse un vide entre la mèche de la bougie précédente et la mèche de la bougie suivante.
""",
            order=1,
            duration="30 min"
        )
        session.add(l4)

        await session.commit()
    print("Academy populated successfully.")

if __name__ == "__main__":
    asyncio.run(populate_academy())
