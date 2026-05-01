"use client";

import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ShieldAlert, Calculator, Info, TrendingUp, AlertTriangle, Loader2 } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { toast } from 'sonner';

export default function RiskPage() {
  // Position Sizing
  const [capital, setCapital] = useState(1000000);
  const [riskPercent, setRiskPercent] = useState(1);
  const [entryPrice, setEntryPrice] = useState(65000);
  const [stopLoss, setStopLoss] = useState(64000);
  const [result, setResult] = useState<any>(null);

  // Kelly
  const [winRate, setWinRate] = useState(0.5);
  const [riskReward, setRiskReward] = useState(2);
  const [kellyResult, setKellyResult] = useState<any>(null);

  // VaR
  const [varResult, setVarResult] = useState<any>(null);
  const [simulating, setSimulating] = useState(false);

  const calculateSize = async () => {
    try {
      const res = await apiClient.post('/risk/position-size', {
        capital,
        risk_percent: riskPercent,
        entry_price: entryPrice,
        stop_loss: stopLoss
      });
      setResult(res.data);
      toast.success("Calcul de position réussi");
    } catch (error) {
      toast.error("Erreur de calcul");
    }
  };

  const calculateKelly = async () => {
    try {
      const res = await apiClient.post('/risk/kelly', {
        win_rate: winRate,
        win_loss_ratio: riskReward
      });
      setKellyResult(res.data);
      toast.success("Critère de Kelly calculé");
    } catch (error) {
      toast.error("Erreur Kelly");
    }
  };

  const simulateVaR = async () => {
    setSimulating(true);
    try {
      const res = await apiClient.post(`/risk/simulate-var?capital=${capital}&volatility=0.02&days=1&iterations=10000`);
      setVarResult(res.data);
      toast.success("Simulation Monte Carlo terminée");
    } catch (error) {
      toast.error("Erreur VaR");
    } finally {
      setSimulating(false);
    }
  };

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="mb-8 text-center md:text-left">
          <h1 className="text-3xl font-bold mb-2">Gestion des Risques</h1>
          <p className="text-muted-foreground">Outils mathématiques avancés pour protéger votre capital.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* Calculator Section */}
            <Card className="border-primary/20 shadow-lg shadow-primary/5">
                <CardHeader>
                <div className="flex items-center gap-2">
                    <Calculator className="w-5 h-5 text-primary" />
                    <CardTitle>Calculateur de Position</CardTitle>
                </div>
                <CardDescription>Déterminez la taille de lot idéale.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                    <Label>Capital ($)</Label>
                    <Input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))} />
                    </div>
                    <div className="space-y-2">
                    <Label>Risque par Trade : {riskPercent}%</Label>
                    <Slider value={[riskPercent]} onValueChange={(val) => setRiskPercent(val[0])} max={5} step={0.1} />
                    </div>
                    <div className="space-y-2">
                    <Label>Entrée ($)</Label>
                    <Input type="number" value={entryPrice} onChange={(e) => setEntryPrice(Number(e.target.value))} />
                    </div>
                    <div className="space-y-2">
                    <Label>Stop Loss ($)</Label>
                    <Input type="number" value={stopLoss} onChange={(e) => setStopLoss(Number(e.target.value))} />
                    </div>
                </div>
                <Button onClick={calculateSize} className="w-full">Calculer</Button>
                {result && (
                    <div className="mt-6 p-4 rounded-lg bg-primary/5 border border-primary/10 grid grid-cols-3 gap-4">
                        <div className="text-center">
                            <p className="text-xs text-muted-foreground">Quantité</p>
                            <p className="text-xl font-bold">{result.quantity}</p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-muted-foreground">Risque $</p>
                            <p className="text-xl font-bold text-red-500">${result.risk_amount}</p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-muted-foreground">Notionnel</p>
                            <p className="text-xl font-bold">${result.notional_value.toLocaleString()}</p>
                        </div>
                    </div>
                )}
                </CardContent>
            </Card>

            {/* VaR Section */}
            <Card>
                <CardHeader>
                    <div className="flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-orange-500" />
                        <CardTitle>Simulation Value at Risk (VaR)</CardTitle>
                    </div>
                    <CardDescription>Simulation Monte Carlo (10,000 itérations) sur 24h.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <Button variant="outline" className="w-full" onClick={simulateVaR} disabled={simulating}>
                        {simulating ? <Loader2 className="animate-spin mr-2" /> : "Lancer la Simulation Monte Carlo"}
                    </Button>
                    {varResult && (
                        <div className="grid grid-cols-3 gap-4 p-4 bg-muted/50 rounded-lg animate-in fade-in zoom-in-95">
                            <div className="text-center">
                                <p className="text-xs text-muted-foreground">VaR (95%)</p>
                                <p className="text-lg font-bold text-orange-500">-${varResult.var_95.toLocaleString()}</p>
                            </div>
                            <div className="text-center">
                                <p className="text-xs text-muted-foreground">VaR (99%)</p>
                                <p className="text-lg font-bold text-red-500">-${varResult.var_99.toLocaleString()}</p>
                            </div>
                            <div className="text-center">
                                <p className="text-xs text-muted-foreground">Pire Cas</p>
                                <p className="text-lg font-bold text-red-700">-${varResult.worst_case.toLocaleString()}</p>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>
          </div>

          <div className="space-y-8">
            <Card className="bg-amber-500/5 border-amber-500/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2 text-amber-500">
                  <ShieldAlert className="w-4 h-4" /> Règle de Survie
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-amber-200/80">
                  Risquez max 1% par trade. Cela vous permet d'encaisser 100 pertes consécutives avant de vider le compte.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-emerald-500" />
                  <CardTitle className="text-base">Optimisation de Kelly</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                    <Label className="text-xs text-muted-foreground">Win Rate Estimé ({Math.round(winRate*100)}%)</Label>
                    <Slider value={[winRate * 100]} onValueChange={v => setWinRate(v[0]/100)} max={100} />
                </div>
                <div className="space-y-2">
                    <Label className="text-xs text-muted-foreground">Ratio Risque/Récompense (1:{riskReward})</Label>
                    <Slider value={[riskReward]} onValueChange={v => setRiskReward(v[0])} min={1} max={10} step={0.5} />
                </div>
                <Button variant="secondary" className="w-full text-xs" onClick={calculateKelly}>Calculer Kelly</Button>
                {kellyResult && (
                    <div className="pt-4 border-t text-center">
                        <p className="text-2xl font-bold text-emerald-500">{kellyResult.kelly_percent}%</p>
                        <p className="text-[10px] text-muted-foreground">Fraction de capital optimale</p>
                        <p className="text-xs mt-2 font-medium">Recommandé (Half-Kelly) : {kellyResult.suggested_risk}%</p>
                    </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
