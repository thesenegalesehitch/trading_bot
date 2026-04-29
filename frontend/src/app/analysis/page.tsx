"use client";

import { useState, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loader2, Search, TrendingDown, TrendingUp, Zap, Target } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { toast } from 'sonner';
import { TradingChart } from '@/components/TradingChart';
import { useExperienceLevel } from '@/lib/experience';
import { MarketReplay } from '@/components/MarketReplay';

export default function AnalysisPage() {
  const [symbol, setSymbol] = useState('BTC-USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('ict');
  const [replayTime, setReplayTime] = useState<string | null>(null);

  const { level, updateLevel, translate } = useExperienceLevel();

  const fetchData = async () => {
    setLoading(true);
    try {
      // 1. Fetch Market Data for Chart
      const replayParam = replayTime ? `&end_time=${replayTime}` : '';
      const marketRes = await apiClient.get(`/market/klines/${symbol}?interval=${timeframe}${replayParam}`);
      const formattedChart = marketRes.data.map((d: any) => ({
        time: new Date(d.timestamp).getTime() / 1000,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      })).sort((a:any, b:any) => a.time - b.time);
      setChartData(formattedChart);

      // 2. Fetch Institutional Analysis
      const analysisRes = await apiClient.get(`/analysis/ict/${symbol}?timeframe=${timeframe}${replayParam}`);
      setAnalysisData(analysisRes.data);
      
      toast.success("Analyse mise à jour");
    } catch (error) {
      toast.error("Échec de la récupération des données");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [symbol, timeframe, replayTime]);

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Analyse Institutionnelle</h1>
            <p className="text-muted-foreground">
              {level === 'beginner' 
                ? "Apprenez à repérer où les banques achètent et vendent sans jargon complexe."
                : "Repérez les empreintes algorithmiques des banques (ICT / SMC)."}
            </p>
          </div>
          <div className="flex items-center gap-6">
            <MarketReplay onTimeChange={setReplayTime} />
            <div className="flex bg-card rounded-lg p-1 border text-[10px] font-bold">
              <button 
                onClick={() => updateLevel('beginner')}
                className={`px-3 py-1.5 rounded-md transition-all ${level === 'beginner' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-muted'}`}
              >
                DÉBUTANT
              </button>
              <button 
                onClick={() => updateLevel('expert')}
                className={`px-3 py-1.5 rounded-md transition-all ${level === 'expert' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-muted'}`}
              >
                EXPERT
              </button>
            </div>
            <div className="flex gap-2 bg-card p-2 rounded-xl border shadow-sm">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-3 text-muted-foreground" />
                <Input 
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="pl-9 w-32 uppercase h-10"
                  placeholder="BTC-USD"
                />
              </div>
              <select 
                value={timeframe} 
                onChange={(e) => setTimeframe(e.target.value)}
                className="px-4 rounded-md bg-secondary text-sm border focus:ring-2 focus:ring-ring h-10"
              >
                <option value="15m">15m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
                <option value="1d">1d</option>
              </select>
              <Button onClick={fetchData} disabled={loading} className="h-10">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
                {loading ? "" : "Scanner"}
              </Button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main Chart Section */}
          <div className="lg:col-span-3 space-y-6">
            <Card className="overflow-hidden border-primary/10 shadow-lg">
              <CardHeader className="flex flex-row items-center justify-between bg-muted/5 border-b pb-4">
                <div>
                  <CardTitle>{symbol} — {timeframe}</CardTitle>
                  <CardDescription>Visualisation temps réel avec superposition {level === 'beginner' ? 'pédagogique' : 'SMC'}.</CardDescription>
                </div>
                <div className="flex gap-4 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-blue-500/20 border border-blue-500 rounded" />
                    <span>{translate('FVG', 'Zone de Déséquilibre')}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-purple-500/20 border border-purple-500 rounded" />
                    <span>{translate('Order Block', 'Zone de Rebond')}</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                {loading ? (
                  <div className="h-[500px] flex items-center justify-center bg-card/50">
                    <Loader2 className="w-12 h-12 animate-spin text-primary" />
                  </div>
                ) : (
                  <TradingChart 
                    data={chartData} 
                    markers={analysisData?.setups?.map((s: any) => ({
                      time: chartData[chartData.length - 1]?.time, // Marker on last candle for current setup
                      position: s.direction === 'BULLISH' ? 'belowBar' : 'aboveBar',
                      color: s.direction === 'BULLISH' ? '#10b981' : '#ef4444',
                      shape: s.direction === 'BULLISH' ? 'arrowUp' : 'arrowDown',
                      text: `${translate('ICT', 'Signal')} ${s.direction}`,
                    }))}
                    fvgZones={analysisData?.setups?.flatMap((s: any) => [
                        { price: s.entry, label: translate('Entry FVG', 'Point d\'Entrée'), color: '#3b82f6' },
                        { price: s.stop_loss, label: translate('Stop Loss', 'Protection'), color: '#ef4444' }
                    ])}
                  />
                )}
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
               <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Setups Détectés</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {analysisData?.setups?.map((s:any, i:number) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg border bg-card hover:border-primary/50 transition-all">
                        <div className="flex gap-4 items-center flex-1">
                          {s.direction === 'BULLISH' ? <TrendingUp className="text-emerald-500 w-8 h-8" /> : <TrendingDown className="text-red-500 w-8 h-8" />}
                          <div className="flex-1">
                            <div className="flex justify-between">
                              <p className="font-bold text-lg">{translate(s.direction + ' Setup', s.direction === 'BULLISH' ? 'Opportunité d\'Achat' : 'Opportunité de Vente')}</p>
                              <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">{s.killzone}</span>
                            </div>
                            <p className="text-xs text-blue-400 mt-1 font-medium italic">
                                {level === 'beginner' 
                                    ? (s.direction === 'BULLISH' ? "Les institutions achètent massivement ici." : "Les institutions vendent massivement ici.")
                                    : s.reason}
                            </p>
                            <div className="grid grid-cols-2 gap-2 mt-2 text-[10px] text-muted-foreground bg-muted/30 p-2 rounded">
                                <div>{translate('Liquidité', 'Zone de Stop-Loss')}: <span className="text-foreground">{s.sequence.liquidity_swept}</span></div>
                                <div>{translate('MSS', 'Changement de Tendance')}: <span className="text-foreground">{s.sequence.mss_impulse}</span></div>
                                <div>{translate('Tap HTF', 'Confirmé à Long Terme')}: <span className="text-foreground">{s.sequence.fvg_tap}</span></div>
                                <div>RR: <span className="text-foreground">{s.risk_reward.toFixed(2)}</span></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    {!analysisData?.setups?.length && <p className="text-sm text-center text-muted-foreground py-8">Aucun setup haute probabilité détecté.</p>}
                  </div>
                </CardContent>
               </Card>

               <Card>
                <CardHeader>
                  <CardTitle className="text-lg">{translate('Liquidité & Zones', 'Niveaux Importants')}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between p-2 rounded bg-muted/50 border-l-2 border-l-blue-500">
                      <span className="text-xs">{translate('Dernier FVG Bullish', 'Déséquilibre Acheteur')}</span>
                      <span className="text-xs font-mono font-bold">$64,230 - $64,450</span>
                    </div>
                    <div className="flex justify-between p-2 rounded bg-muted/50 border-l-2 border-l-purple-500">
                      <span className="text-xs">{translate('Order Block Porteur', 'Point de Soutien')}</span>
                      <span className="text-xs font-mono font-bold">$62,100</span>
                    </div>
                    <div className="flex justify-between p-2 rounded bg-muted/50 border-l-2 border-l-amber-500">
                      <span className="text-xs">{translate('Liquidity Pool (BSL)', 'Zone de Liquidité Haute')}</span>
                      <span className="text-xs font-mono font-bold">$68,500</span>
                    </div>
                  </div>
                </CardContent>
               </Card>
            </div>
          </div>

          {/* Right Sidebar - Logic Explainer */}
          <div className="space-y-6">
            <Card className="bg-primary/5 border-primary/10">
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  <Target className="w-4 h-4 text-primary" /> Algorithme ICT
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground leading-relaxed">
                  L'IA scanne en continu la structure du marché pour identifier le **Market Structure Shift (MSS)**. 
                </p>
                <div className="text-[10px] space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span>Étape 1: Liquidity Sweep</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span>Étape 2: Cassure de Structure (MSS)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span>Étape 3: Entrée sur FVG (Dégagement)</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Killzones Actuelles</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="p-2 rounded bg-secondary text-xs flex justify-between">
                  <span>London</span>
                  <span className="text-muted-foreground">07:00-10:00</span>
                </div>
                <div className="p-2 rounded bg-primary/20 text-xs flex justify-between text-primary font-bold border border-primary/20">
                  <span>New York</span>
                  <span>12:00-15:00</span>
                </div>
                <div className="p-2 rounded bg-secondary text-xs flex justify-between">
                  <span>Asian</span>
                  <span className="text-muted-foreground">23:00-02:00</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
