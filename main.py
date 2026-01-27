"""
Quantum Trading System - Alpha Unification Entry Point
=====================================================
Professional quantitative engine for Forex, Crypto and Indices.

Architecture: Clean / Institutional
Layers: Domain, Application, Infrastructure, Shared
"""

import sys
import os
import asyncio

# Boostrap the package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum.engine.orchestrator import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] System shutdown requested by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[CRITICAL] Fatal system error: {e}")
        sys.exit(1)
