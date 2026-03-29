"use client";

import { useState } from 'react';
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

export default function TradePage() {
  const [symbol, setSymbol] = useState('BTC-USD');
  const [quantity, setQuantity] = useState('0.1');
  const [side, setSide] = useState('BUY');
  const [loading, setLoading] = useState(false);
  
  const [positions, setPositions] = useState([
    { id: 1, symbol: 'EURUSD=X', side: 'BUY', qty: 100000, entry: 1.0854, current: 1.0892, pnl: 380, status: 'OPEN' },
    { id: 2, symbol: 'TSLA', side: 'SELL', qty: 50, entry: 210.5, current: 205.1, pnl: 270, status: 'OPEN' },
  ]);

  const handleTrade = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    // Simuler le délai API
    setTimeout(() => {
      const newPos = {
        id: Date.now(),
        symbol: symbol.toUpperCase(),
        side: side,
        qty: parseFloat(quantity),
        entry: 64200.50, // Prix fictif
        current: 64200.50,
        pnl: 0,
        status: 'OPEN'
      };
      setPositions([...positions, newPos]);
      toast.success(`Ordre exécuté : ${side} ${quantity} ${symbol}`);
      setLoading(false);
    }, 800);
  };

  const closePosition = (id: number) => {
    toast.info('Clôture de la position en cours...');
    setTimeout(() => {
      setPositions(positions.filter(p => p.id !== id));
      toast.success('Position clôturée avec succès.');
    }, 600);
  };

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Trading Démo</h1>
          <p className="text-muted-foreground">Exécutez vos stratégies en environnement simulé (Fonds: $1,015,420).</p>
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
                          <TableCell>{pos.qty}</TableCell>
                          <TableCell>${pos.entry.toLocaleString()}</TableCell>
                          <TableCell>${pos.current.toLocaleString()}</TableCell>
                          <TableCell className={`text-right font-bold ${pos.pnl >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                            {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toLocaleString()}
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
