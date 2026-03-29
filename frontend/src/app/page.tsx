"use client";

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, BookOpen, CandlestickChart, ShieldAlert } from 'lucide-react';

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="px-6 py-4 border-b flex items-center justify-between sticky top-0 bg-background/80 backdrop-blur-md z-50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <CandlestickChart className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-xl font-bold tracking-tight">Quantum Trading</span>
        </div>
        <nav className="flex items-center gap-4">
          <Link href="/login" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
            Connexion
          </Link>
          <Button onClick={() => window.location.href='/register'}>Commencer</Button>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="flex-1">
        <section className="py-24 px-6 md:py-32 flex flex-col items-center text-center max-w-5xl mx-auto">
          <div className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80 mb-6">
            ✨ Version 2.0 - Plateforme Référence Mondiale
          </div>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 bg-gradient-to-br from-foreground to-muted-foreground bg-clip-text text-transparent">
            Maîtrisez les marchés financiers avec Quantum
          </h1>
          <p className="text-xl text-muted-foreground mb-10 max-w-3xl">
            Du débutant à l'expert. Identifiez les empreintes institutionnelles avec nos algorithmes (ICT, SMC, Wyckoff), pratiquez sans risque avec 1M$ virtuel, et formez-vous.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 w-full justify-center">
            <Button size="lg" className="h-12 px-8 text-base" onClick={() => window.location.href='/register'}>
              Ouvrir un compte démo
            </Button>
            <Button size="lg" variant="outline" className="h-12 px-8 text-base" onClick={() => window.location.href='/learn'}>
              Explorer les cours
            </Button>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 px-6 bg-muted/50 border-t">
          <div className="max-w-6xl mx-auto grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="bg-background">
              <CardHeader>
                <CandlestickChart className="w-10 h-10 text-primary mb-2" />
                <CardTitle>Analyse Institutionnelle</CardTitle>
                <CardDescription>
                  Détection automatique de Fair Value Gaps, Order Blocks, Liquidity Sweeps (ICT/SMC).
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-background">
              <CardHeader>
                <Activity className="w-10 h-10 text-primary mb-2" />
                <CardTitle>Simulateur Trading</CardTitle>
                <CardDescription>
                  Paper trading en temps réel avec solde virtuel de 1 000 000 $. Testez vos stratégies sans risque.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-background">
              <CardHeader>
                <BookOpen className="w-10 h-10 text-primary mb-2" />
                <CardTitle>Académie Complète</CardTitle>
                <CardDescription>
                  10 cours structurés, de l'indépendance financière à la maîtrise des profils de marché avancés.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="bg-background">
              <CardHeader>
                <ShieldAlert className="w-10 h-10 text-primary mb-2" />
                <CardTitle>Gestion du Risque (VaR)</CardTitle>
                <CardDescription>
                  Calcul de la Value at Risk (historique, paramétrique, Monte Carlo) pour un portefeuille robuste.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </section>
      </main>
      
      {/* Footer */}
      <footer className="py-8 px-6 border-t md:flex items-center justify-between text-muted-foreground text-sm">
        <p>© 2026 Quantum Trading System by Alexandre Albert Ndour. Tous droits réservés.</p>
        <div className="flex gap-4 mt-4 md:mt-0">
          <Link href="/terms" className="hover:text-foreground">Conditions</Link>
          <Link href="/privacy" className="hover:text-foreground">Confidentialité</Link>
        </div>
      </footer>
    </div>
  );
}
