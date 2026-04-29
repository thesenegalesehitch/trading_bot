"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { CandlestickChart, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { toast } from 'sonner';
import { apiClient } from '@/lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    username: '', // email
    password: ''
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // FastAPI OAuth2PasswordRequestForm requires form-urlencoded data
      const params = new URLSearchParams();
      params.append('username', formData.username);
      params.append('password', formData.password);

      const response = await apiClient.post('/auth/login', params, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      
      localStorage.setItem('token', response.data.access_token);
      toast.success('Connexion réussie', { description: 'Redirection vers le dashboard...' });
      router.push('/dashboard');
    } catch (error: any) {
      toast.error('Erreur', { 
        description: error.response?.data?.detail || 'Identifiants invalides.' 
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-muted/30 flex items-center justify-center p-4">
      <div className="absolute top-4 left-4">
        <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <CandlestickChart className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-bold">Quantum</span>
        </Link>
      </div>

      <Card className="w-full max-w-md shadow-lg border-primary/10">
        <CardHeader className="space-y-1 text-center">
          <CardTitle className="text-2xl font-bold">Heureux de vous revoir</CardTitle>
          <CardDescription>Entrez vos identifiants pour accéder à votre espace.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input 
                id="email" 
                type="email" 
                placeholder="trader@quantum.com" 
                required 
                value={formData.username}
                onChange={(e) => setFormData({...formData, username: e.target.value})}
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Mot de passe</Label>
                {/* <Link href="#" className="text-xs text-primary hover:underline">Mot de passe oublié ?</Link> */}
              </div>
              <Input 
                id="password" 
                type="password" 
                required 
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
              />
            </div>
            <Button className="w-full" type="submit" disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Se connecter
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex justify-center border-t p-4 text-sm">
          <span className="text-muted-foreground mr-1">Nouveau sur Quantum ?</span>
          <Link href="/register" className="text-primary font-medium hover:underline">
            Créer un compte ($1M virtuel offert)
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
