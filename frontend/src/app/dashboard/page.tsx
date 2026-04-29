"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, DollarSign, PieChart, TrendingUp, CandlestickChart, Loader2 } from 'lucide-react';
import { authApi, tradingApi, apiClient } from '@/lib/api';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<{ full_name: string } | null>(null);
  const [account, setAccount] = useState<{ balance: number, created_at: string } | null>(null);
  const [positions, setPositions] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [profileRes, accountRes, positionsRes, historyRes] = await Promise.all([
          authApi.getMe(),
          tradingApi.getAccount(),
          tradingApi.getPositions(),
          apiClient.get('/trading/history')
        ]);
        
        setUser(profileRes.data);
        setAccount(accountRes.data);
        setPositions(positionsRes.data);
        
        // Formater l'historique pour le graphique
        const formattedHistory = historyRes.data.map((h: any) => ({
          date: format(new Date(h.timestamp), 'dd MMM', { locale: fr }),
          value: h.balance
        }));
        setHistory(formattedHistory);
      } catch (error) {
        console.error("Failed to fetch dashboard data", error);
        router.push('/login');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [router]);

  if (loading) return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
      <Loader2 className="w-10 h-10 animate-spin text-primary" />
      <p className="text-muted-foreground animate-pulse">Synchronisation de vos données quantiques...</p>
    </div>
  );

  const initialBalance = 1000000;
  const currentBalance = account?.balance || initialBalance;
  const totalPnL = currentBalance - initialBalance;
  const pnlPercent = (totalPnL / initialBalance) * 100;

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Tableau de Bord</h1>
            <p className="text-muted-foreground">Bienvenue, {user?.full_name}. Aperçu de vos performances institutionnelles.</p>
          </div>
          <div className="bg-card px-4 py-2 rounded-full border shadow-sm text-sm font-medium flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Marché Ouvert
          </div>
        </div>

        {/* KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="border-l-4 border-l-primary">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Capital Net (Equity)</CardTitle>
              <DollarSign className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${currentBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <p className={`text-xs flex items-center mt-1 font-medium ${totalPnL >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                <TrendingUp className={`h-3 w-3 mr-1 ${totalPnL < 0 ? 'rotate-180' : ''}`} /> 
                {totalPnL >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}% depuis l'origine
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Positions Actives</CardTitle>
              <Activity className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{positions.length}</div>
              <p className="text-xs text-muted-foreground mt-1">
                Exposition actuelle sur le marché
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Risque VaR (95%)</CardTitle>
              <PieChart className="h-4 w-4 text-orange-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-500">
                -${(currentBalance * 0.012).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Perte max. probable / 24h
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Score Académique</CardTitle>
              <CandlestickChart className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Niveau 1</div>
              <p className="text-xs text-muted-foreground mt-1">
                42% du cursus complété
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="col-span-2 shadow-md border-primary/5">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Courbe de Performance</CardTitle>
                <p className="text-sm text-muted-foreground">Evolution de votre solde en temps réel.</p>
              </div>
              <div className="flex gap-2">
                {['1J', '1S', '1M', 'TOUT'].map((p) => (
                  <button key={p} className={`text-xs px-2 py-1 rounded ${p === 'TOUT' ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'}`}>{p}</button>
                ))}
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[350px] w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history.length > 0 ? history : [{date: 'Démarrage', value: 1000000}]}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--muted)/0.2)" />
                    <XAxis 
                      dataKey="date" 
                      stroke="hsl(var(--muted-foreground))" 
                      fontSize={11} 
                      tickLine={false} 
                      axisLine={false} 
                      dy={10}
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))" 
                      fontSize={11} 
                      tickLine={false} 
                      axisLine={false} 
                      tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                      domain={['dataMin - 1000', 'dataMax + 1000']}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--card))', 
                        borderColor: 'hsl(var(--border))',
                        borderRadius: '12px',
                        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
                      }}
                      itemStyle={{ color: 'hsl(var(--primary))', fontWeight: 'bold' }}
                      labelStyle={{ color: 'hsl(var(--muted-foreground))', marginBottom: '4px' }}
                      formatter={(value: any) => [`$${Number(value).toLocaleString()}`, 'Balance']}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="hsl(var(--primary))" 
                      fillOpacity={1} 
                      fill="url(#colorValue)" 
                      strokeWidth={3}
                      animationDuration={1500}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
          
          <Card className="shadow-md border-primary/5">
            <CardHeader>
              <CardTitle>Journal d'Exécution</CardTitle>
              <p className="text-sm text-muted-foreground">Flux d'activités récentes.</p>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {positions.length > 0 ? (
                  positions.map((pos) => (
                    <div key={pos.id} className="flex items-center justify-between p-2 rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full bg-blue-500`} />
                        <div>
                          <p className="text-sm font-medium">Position Ouverte <span className="text-primary ml-1">{pos.symbol}</span></p>
                          <p className="text-xs text-muted-foreground">{pos.side} @ ${pos.price.toLocaleString()}</p>
                        </div>
                      </div>
                      <span className="text-xs font-mono bg-muted px-2 py-1 rounded">
                        {pos.quantity} units
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12">
                    <p className="text-sm text-muted-foreground">Aucune activité récente.</p>
                    <p className="text-xs text-muted-foreground mt-2">Commencez à trader pour voir vos stats s'animer.</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
