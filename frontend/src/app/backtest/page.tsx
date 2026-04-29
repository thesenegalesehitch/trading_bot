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

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="md:col-span-1">
                <CardHeader>
                    <CardTitle>Configuration</CardTitle>
                    <CardDescription>Définissez vos paramètres de test.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="p-4 rounded-lg bg-muted/50 border space-y-2">
                        <p className="text-sm font-bold">Symbole: BTC-USD</p>
                        <p className="text-sm">Timeframe: 1h</p>
                        <p className="text-sm">Période: 2024-01-01 -> 2024-04-29</p>
                    </div>
                    <Button className="w-full">
                        <Play className="w-4 h-4 mr-2" />
                        Lancer le Backtest
                    </Button>
                </CardContent>
            </Card>

            <Card className="md:col-span-2">
                <CardHeader>
                    <CardTitle>Historique des Runs</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground border-2 border-dashed rounded-lg">
                        <History className="w-12 h-12 mb-4 opacity-20" />
                        <p>Aucun backtest enregistré.</p>
                    </div>
                </CardContent>
            </Card>
        </div>
      </main>
    </div>
  );
}
