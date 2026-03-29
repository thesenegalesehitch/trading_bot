"use client";

import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loader2, Search, ArrowRight, ShieldAlert, TrendingDown, TrendingUp } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { toast } from 'sonner';

export default function AnalysisPage() {
  const [symbol, setSymbol] = useState('BTC-USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('ict');

  const handleScan = async () => {
    if (!symbol) return;
    setLoading(true);
    try {
      // Pour l'exemple, on appelle l'API correspondante à l'onglet actif.
      const endpoint = `/analysis/${activeTab}/${symbol}?timeframe=${timeframe}`;
      const res = await apiClient.get(endpoint);
      setData(res.data);
      toast.success('Scan terminé', { description: `Analyse structurée pour ${symbol}` });
    } catch (error: any) {
      toast.error('Erreur de scan', { description: error.response?.data?.detail || 'Données insuffisantes' });
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Analyse Quantitative</h1>
            <p className="text-muted-foreground">Détection d'empreintes institutionnelles (ICT, SMC, Wyckoff).</p>
          </div>
          <div className="flex gap-2 bg-card p-2 rounded-lg border shadow-sm">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-3 text-muted-foreground" />
              <Input 
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="pl-9 w-32 uppercase"
                placeholder="BTC-USD"
              />
            </div>
            <select 
              value={timeframe} 
              onChange={(e) => setTimeframe(e.target.value)}
              className="px-3 rounded-md bg-secondary text-sm border focus:ring-2 focus:ring-ring"
            >
              <option value="15m">15m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
            <Button onClick={handleScan} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : "Scanner"}
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-3 mb-8">
            <TabsTrigger value="ict">ICT Setup</TabsTrigger>
            <TabsTrigger value="smc">SMC Structure</TabsTrigger>
            <TabsTrigger value="wyckoff">Wyckoff Phase</TabsTrigger>
          </TabsList>

          <TabsContent value="ict" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="col-span-2">
                <CardHeader>
                  <CardTitle>Configurations Inner Circle Trader (ICT)</CardTitle>
                  <CardDescription>Recherche de Liquidity Sweep → MSS → FVG.</CardDescription>
                </CardHeader>
                <CardContent>
                  {!data ? (
                    <div className="h-40 flex items-center justify-center border-2 border-dashed rounded-lg text-muted-foreground">
                      Lancez un scan pour voir les opportunités
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {data.setups?.length === 0 ? (
                        <p className="text-muted-foreground text-center py-4">Aucun setup ICT valide trouvé.</p>
                      ) : (
                        data.setups?.map((setup: any, i: number) => (
                          <div key={i} className="p-4 border rounded-lg bg-card flex justify-between items-center">
                            <div className="flex gap-4 items-center">
                              {setup.direction === 'BULLISH' ? <TrendingUp className="text-emerald-500 w-8 h-8" /> : <TrendingDown className="text-red-500 w-8 h-8" />}
                              <div>
                                <p className="font-bold text-lg">{setup.direction} Setup</p>
                                <p className="text-sm text-muted-foreground">Entrée FVG à ${setup.entry?.toFixed(2)}</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-sm font-medium">Confiance: {(setup.confidence * 100).toFixed(0)}%</p>
                              <p className="text-xs text-muted-foreground">R/R: {setup.risk_reward?.toFixed(2)}</p>
                            </div>
                            <Button size="sm" variant="outline">Trader</Button>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Killzones Actives</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center p-2 rounded bg-secondary">
                      <span>London</span>
                      <span className="text-xs font-mono text-muted-foreground">07:00-10:00 GMT</span>
                    </div>
                    <div className="flex justify-between items-center p-2 rounded bg-primary/20 text-primary border border-primary/30">
                      <span>New York</span>
                      <span className="text-xs font-mono">12:00-15:00 GMT</span>
                    </div>
                    <div className="flex justify-between items-center p-2 rounded bg-secondary">
                      <span>Asian</span>
                      <span className="text-xs font-mono text-muted-foreground">23:00-02:00 GMT</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Fallbacks pour les autres onglets dans cette démo */}
          <TabsContent value="smc">
            <Card>
              <CardHeader>
                <CardTitle>Fair Value Gaps & Order Blocks</CardTitle>
              </CardHeader>
              <CardContent className="h-64 flex items-center justify-center text-muted-foreground flex-col gap-2">
                <ShieldAlert className="w-8 h-8" />
                <p>Analyse de la microstructure SMC.</p>
                {data && activeTab === 'smc' && <pre className="text-xs bg-muted p-4 rounded overflow-auto max-w-full mt-4">{JSON.stringify(data, null, 2)}</pre>}
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="wyckoff">
            <Card>
              <CardHeader>
                <CardTitle>Cycle d'Accumulation / Distribution</CardTitle>
              </CardHeader>
              <CardContent className="h-64 flex justify-center text-muted-foreground flex-col gap-2">
                {data && activeTab === 'wyckoff' ? (
                  <div className="text-center">
                    <p className="text-2xl font-bold uppercase mb-4 text-primary">{data.phase}</p>
                    <p className="mb-2">Derniers évènements :</p>
                    {data.events?.map((e: any, i:number) => <p key={i} className="text-sm font-mono">{e.event} @ ${e.price?.toFixed(2)}</p>)}
                  </div>
                ) : (
                  <div className="text-center">Lancez le scan Wyckoff...</div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
