"use client";

import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { History, Play, FileJson } from 'lucide-react';

export default function BacktestPage() {
  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <h1 className="text-3xl font-bold mb-8">Moteur de Backtesting</h1>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            <Card className="lg:col-span-1 border-primary/20">
                <CardHeader>
                    <CardTitle>Configuration</CardTitle>
                    <CardDescription>Paramétrez votre test historique.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <label className="text-xs font-medium uppercase text-muted-foreground">Actif</label>
                        <select className="w-full p-2 rounded-md border bg-background">
                            <option>BTC-USD</option>
                            <option>EURUSD=X</option>
                            <option>AAPL</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="text-xs font-medium uppercase text-muted-foreground">Période</label>
                        <div className="grid grid-cols-2 gap-2">
                            <input type="date" className="p-2 rounded-md border bg-background text-xs" defaultValue="2024-01-01" />
                            <input type="date" className="p-2 rounded-md border bg-background text-xs" defaultValue="2024-04-30" />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label className="text-xs font-medium uppercase text-muted-foreground">Capital Initial</label>
                        <input type="number" className="w-full p-2 rounded-md border bg-background" defaultValue="10000" />
                    </div>
                    <Button className="w-full bg-primary hover:bg-primary/90 shadow-lg shadow-primary/20">
                        <Play className="w-4 h-4 mr-2" />
                        Lancer le Backtest
                    </Button>
                </CardContent>
            </Card>

            <Card className="lg:col-span-3">
                <CardHeader>
                    <CardTitle>Rapport de Performance</CardTitle>
                    <CardDescription>Analyse détaillée des trades passés.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col items-center justify-center py-20 text-muted-foreground border-2 border-dashed rounded-xl bg-muted/5">
                        <History className="w-16 h-16 mb-4 opacity-10" />
                        <p className="font-medium">Aucun backtest exécuté sur cette période.</p>
                        <p className="text-xs mt-2">Sélectionnez vos paramètres et cliquez sur "Lancer".</p>
                    </div>
                </CardContent>
            </Card>
        </div>
      </main>
    </div>
  );
}
