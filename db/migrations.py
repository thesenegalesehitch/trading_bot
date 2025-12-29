"""
Database migration scripts for Quantum Trading System.

Handles table creation, data migration, and schema updates.
"""

import sys
import os
from sqlalchemy import create_engine, text
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config
from db.models import Base

logger = logging.getLogger(__name__)


def create_tables(engine):
    """Crée toutes les tables de la base de données."""
    try:
        logger.info("Création des tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Tables créées avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur création tables: {e}")
        return False


def drop_tables(engine):
    """Supprime toutes les tables (utiliser avec précaution)."""
    try:
        logger.warning("Suppression des tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("Tables supprimées")
        return True
    except Exception as e:
        logger.error(f"Erreur suppression tables: {e}")
        return False


def insert_initial_data(engine):
    """Insère les données initiales (symboles, etc.)."""
    try:
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()

        # Insérer les symboles actifs
        from db.models import Symbol

        symbols_data = []
        for symbol in config.symbols.ACTIVE_SYMBOLS:
            display_name = config.symbols.DISPLAY_NAMES.get(symbol, symbol)
            asset_class = "forex" if symbol.endswith("=X") else "crypto" if "USD" in symbol and len(symbol) <= 10 else "equity"
            symbols_data.append({
                'symbol': symbol,
                'name': display_name,
                'asset_class': asset_class,
                'active': True
            })

        for data in symbols_data:
            # Vérifier si existe déjà
            existing = session.query(Symbol).filter_by(symbol=data['symbol']).first()
            if not existing:
                symbol_obj = Symbol(**data)
                session.add(symbol_obj)

        session.commit()
        logger.info(f"Données initiales insérées: {len(symbols_data)} symboles")
        session.close()
        return True

    except Exception as e:
        logger.error(f"Erreur insertion données initiales: {e}")
        return False


def migrate_historical_data(engine):
    """Migre les données historiques existantes vers la base."""
    try:
        logger.info("Migration des données historiques...")

        # Cette fonction migrerait les données depuis les fichiers locaux
        # vers la base de données PostgreSQL

        # Exemple: migrer les données de market_data depuis les fichiers pickle/csv
        # data_dir = config.system.DATA_DIR
        # for symbol in config.symbols.ACTIVE_SYMBOLS:
        #     # Charger les données locales et les insérer en base

        logger.info("Migration terminée (implémentation placeholder)")
        return True

    except Exception as e:
        logger.error(f"Erreur migration données historiques: {e}")
        return False


def run_migrations():
    """Exécute toutes les migrations."""
    # Créer le moteur de base de données
    engine = create_engine(config.database.postgres_url, echo=False)

    success = True

    # Créer les tables
    if not create_tables(engine):
        success = False

    # Insérer données initiales
    if not insert_initial_data(engine):
        success = False

    # Migrer données historiques (optionnel)
    # if not migrate_historical_data(engine):
    #     success = False

    if success:
        logger.info("Toutes les migrations terminées avec succès")
    else:
        logger.error("Échec des migrations")

    return success


def reset_database():
    """Remet à zéro la base de données (ATTENTION: destructive)."""
    engine = create_engine(config.database.postgres_url, echo=False)

    # Supprimer et recréer
    drop_tables(engine)
    create_tables(engine)
    insert_initial_data(engine)

    logger.info("Base de données remise à zéro")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Database migrations')
    parser.add_argument('action', choices=['migrate', 'reset', 'create', 'drop'],
                       help='Action à effectuer')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.action == 'migrate':
        run_migrations()
    elif args.action == 'reset':
        confirm = input("ATTENTION: Cela va supprimer toutes les données. Continuer? (yes/no): ")
        if confirm.lower() == 'yes':
            reset_database()
        else:
            print("Annulé")
    elif args.action == 'create':
        engine = create_engine(config.database.postgres_url, echo=False)
        create_tables(engine)
    elif args.action == 'drop':
        confirm = input("ATTENTION: Cela va supprimer toutes les tables. Continuer? (yes/no): ")
        if confirm.lower() == 'yes':
            engine = create_engine(config.database.postgres_url, echo=False)
            drop_tables(engine)
        else:
            print("Annulé")