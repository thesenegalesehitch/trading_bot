"use client";

import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ShieldAlert, Calculator, Info, TrendingUp, AlertTriangle } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { toast } from 'sonner';

export default function RiskPage() {
  const [capital, setCapital] = useState(1000000);
  const [riskPercent, setRiskPercent] = useState(1);
  const [entryPrice, setEntryPrice] = useState(65000);
  const [stopLoss, setStopLoss] = useState(64000);
  const [result, setResult] = useState<any>(null);

  const calculateSize = async () => {
    try {
      const res = await apiClient.post('/risk/position-size', {
        capital,
        risk_percent: riskPercent,
        entry_price: entryPrice,
        stop_loss: stopLoss
      });
      setResult(res.data);
      toast.success("Calcul réussi");
    } catch (error) {
      toast.error("Erreur de calcul");
    }
  };

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Gestion des Risques</h1>
          <p className="text-muted-foreground">Protégez votre capital avec des outils mathématiques avancés.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Calculator Section */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Calculator className="w-5 h-5 text-primary" />
                <CardTitle>Calculateur de Position</CardTitle>
              </div>
              <CardDescription>Déterminez la taille idéale de votre lot selon votre tolérance au risque.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label>Capital Total ($)</Label>
                  <Input 
                    type="number" 
                    value={capital} 
                    onChange={(e) => setCapital(Number(e.target.value))} 
                    className="h-12 text-lg"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Risque par Trade (%) : {riskPercent}%</Label>
                  <Slider 
                    value={[riskPercent]} 
                    onValueChange={(val) => setRiskPercent(val[0])} 
                    max={5} 
                    step={0.1} 
                    className="py-4"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Prix d'Entrée ($)</Label>
                  <Input 
                    type="number" 
                    value={entryPrice} 
                    onChange={(e) => setEntryPrice(Number(e.target.value))} 
                    className="h-12"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Stop Loss ($)</Label>
                  <Input 
                    type="number" 
                    value={stopLoss} 
                    onChange={(e) => setStopLoss(Number(e.target.value))} 
                    className="h-12"
                  />
                </div>
              </div>

              <Button onClick={calculateSize} className="w-full h-12 text-lg">
                Calculer la Taille de Position
              </Button>

              {result && (
                <div className="mt-8 p-6 rounded-xl bg-primary/5 border border-primary/20 grid grid-cols-1 md:grid-cols-3 gap-6 animate-in fade-in zoom-in-95">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-1">Quantité à Acheter</p>
                    <p className="text-3xl font-bold text-primary">{result.quantity}</p>
                    <p className="text-xs text-muted-foreground mt-1">unités / lots</p>
                  </div>
                  <div className="text-center border-x border-primary/10">
                    <p className="text-sm text-muted-foreground mb-1">Risque Monétaire</p>
                    <p className="text-3xl font-bold text-red-500">${result.risk_amount.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground mt-1">perte max. si SL touché</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground mb-1">Valeur Notionnelle</p>
                    <p className="text-3xl font-bold">${result.notional_value.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground mt-1">exposition totale</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Education / Info Section */}
          <div className="space-y-6">
            <Card className="bg-amber-500/10 border-amber-500/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2 text-amber-500">
                  <ShieldAlert className="w-4 h-4" /> La Règle des 1%
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-amber-200/80 leading-relaxed">
                  Ne risquez jamais plus de 1% de votre capital sur un seul trade. Cela vous permet d'encaisser 100 pertes consécutives avant de tout perdre.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Info className="w-4 h-4 text-blue-500" /> Qu'est-ce que la VaR ?
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground">
                  La Value at Risk (VaR) est utilisée par les banques pour estimer le risque de marché. 
                </p>
                <div className="p-3 bg-muted rounded text-[10px] font-mono">
                  VaR = Capital * (Volatilité * Confidence * sqrt(Temps))
                </div>
                <Button variant="outline" size="sm" className="w-full text-xs">Simuler une VaR Monte Carlo</Button>
              </CardContent>
            </Card>

            <Card className="border-emerald-500/20 bg-emerald-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2 text-emerald-500">
                  <TrendingUp className="w-4 h-4" /> Kelly Criterion
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground mb-4">
                  Optimisez mathématiquement la taille de vos paris pour maximiser la croissance à long terme.
                </p>
                <div className="space-y-3">
                  <div className="flex justify-between text-xs">
                    <span>Win Rate estimé</span>
                    <span className="font-bold">60%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>Ratio Risk/Reward</span>
                    <span className="font-bold">2.5</span>
                  </div>
                  <div className="pt-2 border-t border-emerald-500/10">
                    <p className="text-center font-bold text-emerald-500">Fraction suggérée : 44%</p>
                    <p className="text-[10px] text-center text-muted-foreground mt-1">(Moitié de Kelly recommandé pour sécurité)</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
