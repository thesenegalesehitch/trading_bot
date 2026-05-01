import sys
import os
import sqlite3

# Chemin vers la DB
db_path = "quantum.db"

def fix_schema():
    print(f"🔧 Correction du schéma pour {db_path}...")
    
    if not os.path.exists(db_path):
        print("Base de données non trouvée. Elle sera créée par init_db.py")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Vérifier si la colonne image_url existe dans courses
    cursor.execute("PRAGMA table_info(courses)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "image_url" not in columns:
        print("Ajout de la colonne image_url à la table courses...")
        cursor.execute("ALTER TABLE courses ADD COLUMN image_url VARCHAR(255)")
    
    # Vérifier si la table trading_journal existe
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_journal'")
    if not cursor.fetchone():
        print("La table trading_journal est manquante. Re-exécution de init_db.py recommandée.")
    
    conn.commit()
    conn.close()
    print("✅ Schéma corrigé.")

if __name__ == "__main__":
    fix_schema()
