"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, DollarSign, PieChart, TrendingUp, CandlestickChart } from 'lucide-react';
import { apiClient } from '@/lib/api';

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<{ full_name: string } | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Mock data for the chart pending real portfolio history from backend
  const performanceData = [
    { date: 'Lun', value: 1000000 },
    { date: 'Mar', value: 1002500 },
    { date: 'Mer', value: 1001200 },
    { date: 'Jeu', value: 1008000 },
    { date: 'Ven', value: 1006500 },
    { date: 'Sam', value: 1012000 },
    { date: 'Dim', value: 1015420 },
  ];

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await apiClient.get('/auth/me');
        setUser(res.data);
      } catch (error) {
        // Rediriger vers le login si le token est invalide
        router.push('/login');
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, [router]);

  if (loading) return <div className="min-h-screen bg-background flex items-center justify-center">Chargement...</div>;

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      
      <main className="flex-1 p-8 overflow-y-auto">
        <h1 className="text-3xl font-bold mb-2">Bonjour, {user?.full_name}</h1>
        <p className="text-muted-foreground mb-8">Voici l'aperçu de votre portefeuille virtuel Quantum.</p>

        {/* KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Solde Démo Total</CardTitle>
              <DollarSign className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">$1,015,420.00</div>
              <p className="text-xs text-emerald-500 flex items-center mt-1">
                <TrendingUp className="h-3 w-3 mr-1" /> +1.54% depuis l'ouverture
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Positions Ouvertes</CardTitle>
              <Activity className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">4</div>
              <p className="text-xs text-muted-foreground mt-1">
                2 Actions, 1 Crypto, 1 Forex
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Value at Risk (95%)</CardTitle>
              <PieChart className="h-4 w-4 text-orange-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">-$12,450.00</div>
              <p className="text-xs text-muted-foreground mt-1">
                Risque calculé (Historique)
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Win Rate (30j)</CardTitle>
              <CandlestickChart className="h-4 w-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">68.5%</div>
              <p className="text-xs text-muted-foreground mt-1">
                +4.2% ce mois-ci
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="col-span-2">
            <CardHeader>
              <CardTitle>Performance du Portefeuille</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={performanceData}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                    <XAxis 
                      dataKey="date" 
                      stroke="hsl(var(--muted-foreground))" 
                      fontSize={12} 
                      tickLine={false} 
                      axisLine={false} 
                    />
                    <YAxis 
                      stroke="hsl(var(--muted-foreground))" 
                      fontSize={12} 
                      tickLine={false} 
                      axisLine={false} 
                      tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                      domain={['dataMin - 5000', 'dataMax + 5000']}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                      itemStyle={{ color: 'hsl(var(--foreground))' }}
                      formatter={(value: any) => [`$${Number(value).toLocaleString()}`, 'Portfolio']}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="hsl(var(--primary))" 
                      fillOpacity={1} 
                      fill="url(#colorValue)" 
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Dernières Activités</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {[
                  { id: 1, action: "Achat réussi", sym: "BTC-USD", amt: "$4,500", time: "Il y a 2h", type: "success" },
                  { id: 2, action: "Stop Loss touché", sym: "AAPL", amt: "-$320", time: "Il y a 5h", type: "destructive" },
                  { id: 3, action: "Take Profit", sym: "EURUSD=X", amt: "+$850", time: "Hier", type: "success" },
                  { id: 4, action: "Ordre placé", sym: "TSLA", amt: "$1,200", time: "Hier", type: "default" },
                ].map((act) => (
                  <div key={act.id} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${act.type === 'success' ? 'bg-emerald-500' : act.type === 'destructive' ? 'bg-red-500' : 'bg-blue-500'}`} />
                      <div>
                        <p className="text-sm font-medium">{act.action} <span className="text-muted-foreground ml-1">{act.sym}</span></p>
                        <p className="text-xs text-muted-foreground">{act.time}</p>
                      </div>
                    </div>
                    <span className={`text-sm font-medium ${act.type === 'success' ? 'text-emerald-500' : act.type === 'destructive' ? 'text-red-500' : ''}`}>
                      {act.amt}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
