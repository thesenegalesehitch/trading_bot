"use client";

import { useEffect, useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Loader2, DollarSign } from 'lucide-react';
import { toast } from 'sonner';
import { tradingApi } from '@/lib/api';

export default function TradePage() {
  const [symbol, setSymbol] = useState('BTC-USD');
  const [quantity, setQuantity] = useState('0.1');
  const [side, setSide] = useState('BUY');
  const [loading, setLoading] = useState(false);
  const [account, setAccount] = useState<any>(null);
  
  const [positions, setPositions] = useState<any[]>([]);

  const fetchPositions = async () => {
    try {
      const [posRes, accRes] = await Promise.all([
        tradingApi.getPositions(),
        tradingApi.getAccount()
      ]);
      setPositions(posRes.data);
      setAccount(accRes.data);
    } catch (error) {
      console.error("Failed to fetch positions", error);
    }
  };

  useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, 10000); // Rafraîchir toutes les 10s
    return () => clearInterval(interval);
  }, []);

  const handleTrade = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      await tradingApi.openTrade({
        symbol: symbol,
        side: side,
        quantity: parseFloat(quantity)
      });
      toast.success(`Ordre exécuté : ${side} ${quantity} ${symbol}`);
      fetchPositions();
    } catch (error: any) {
      toast.error('Erreur d\'exécution', { 
        description: error.response?.data?.detail || 'Impossible d\'exécuter l\'ordre.' 
      });
    } finally {
      setLoading(false);
    }
  };

  const closePosition = async (id: number) => {
    toast.info('Clôture de la position en cours...');
    try {
      await tradingApi.closeTrade(id);
      toast.success('Position clôturée avec succès.');
      fetchPositions();
    } catch (error) {
      toast.error('Erreur lors de la clôture');
    }
  };

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Trading Démo</h1>
          <p className="text-muted-foreground">Exécutez vos stratégies en environnement simulé (Fonds: ${account?.balance.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '...'}).</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Order Entry Form */}
          <Card className="col-span-1 border-primary/20">
            <CardHeader>
              <CardTitle>Passer un Ordre</CardTitle>
              <CardDescription>Marché au comptant (Spot)</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleTrade} className="space-y-4">
                <div className="space-y-2">
                  <Label>Actif (Symbole Yahoo)</Label>
                  <Input 
                    value={symbol} 
                    onChange={e => setSymbol(e.target.value.toUpperCase())}
                    placeholder="ex: AAPL, BTC-USD"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label>Action</Label>
                  <Select value={side} onValueChange={(val) => setSide(val || 'BUY')}>
                    <SelectTrigger>
                      <SelectValue placeholder="Sélectionnez l'action" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BUY" className="text-emerald-500 font-bold">Acheter (Long)</SelectItem>
                      <SelectItem value="SELL" className="text-red-500 font-bold">Vendre (Short)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Quantité</Label>
                  <Input 
                    type="number" 
                    step="0.001"
                    min="0.001"
                    value={quantity} 
                    onChange={e => setQuantity(e.target.value)}
                    required
                  />
                </div>
                <Button 
                  type="submit" 
                  className={`w-full ${side === 'BUY' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-red-600 hover:bg-red-700'}`}
                  disabled={loading}
                >
                  {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                  Exécuter l'ordre
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Open Positions */}
          <Card className="col-span-1 lg:col-span-2">
            <CardHeader>
              <CardTitle>Positions Ouvertes</CardTitle>
              <CardDescription>Suivi en temps réel de votre portefeuille</CardDescription>
            </CardHeader>
            <CardContent>
              {positions.length === 0 ? (
                <div className="text-center py-10 text-muted-foreground">
                  Aucune position ouverte.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbole</TableHead>
                        <TableHead>Sens</TableHead>
                        <TableHead>Quantité</TableHead>
                        <TableHead>Prix Entrée</TableHead>
                        <TableHead>Prix Actuel</TableHead>
                        <TableHead className="text-right">PnL Lattant</TableHead>
                        <TableHead></TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {positions.map((pos) => (
                        <TableRow key={pos.id}>
                          <TableCell className="font-medium">{pos.symbol}</TableCell>
                          <TableCell>
                            <Badge variant={pos.side === 'BUY' ? 'default' : 'destructive'} className={pos.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20' : ''}>
                              {pos.side}
                            </Badge>
                          </TableCell>
                          <TableCell>{pos.quantity}</TableCell>
                          <TableCell>${pos.price.toLocaleString()}</TableCell>
                          <TableCell>--</TableCell>
                          <TableCell className={`text-right font-bold ${pos.pnl >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                            {pos.pnl !== undefined ? (pos.pnl >= 0 ? '+' : '') + pos.pnl.toLocaleString() : '--'}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button variant="ghost" size="sm" onClick={() => closePosition(pos.id)} className="text-red-400 hover:text-red-500 hover:bg-red-400/10 h-8 px-2">
                              Fermer
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
