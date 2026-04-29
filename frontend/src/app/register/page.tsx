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

export default function RegisterPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    password: '',
    confirm_password: ''
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (formData.password !== formData.confirm_password) {
      toast.error('Erreur', { description: 'Les mots de passe ne correspondent pas.'});
      return;
    }

    setLoading(true);
    try {
      await apiClient.post('/auth/register', {
        email: formData.email,
        password: formData.password,
        full_name: formData.full_name
      });
      
      toast.success('Compte créé avec succès', { 
        description: 'Votre portfolio virtuel de 1 000 000 $ vous attend !' 
      });
      
      // Auto-login
      const params = new URLSearchParams();
      params.append('username', formData.email);
      params.append('password', formData.password);
      const loginResp = await apiClient.post('/auth/login', params, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      
      localStorage.setItem('token', loginResp.data.access_token);
      router.push('/dashboard');
    } catch (error: any) {
      toast.error('Erreur', { 
        description: error.response?.data?.detail || 'Impossible de créer le compte.' 
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
          <CardTitle className="text-2xl font-bold">Créer un profil Trader</CardTitle>
          <CardDescription>Accédez au programme éducatif et à 1 000 000 $ de fonds virtuels.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Nom complet</Label>
              <Input 
                id="name" 
                placeholder="John Doe" 
                required 
                value={formData.full_name}
                onChange={(e) => setFormData({...formData, full_name: e.target.value})}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email professionnel</Label>
              <Input 
                id="email" 
                type="email" 
                placeholder="trader@quantum.com" 
                required 
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Nouveau mot de passe</Label>
              <Input 
                id="password" 
                type="password" 
                required 
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirm_password">Confirmer le mot de passe</Label>
              <Input 
                id="confirm_password" 
                type="password" 
                required 
                value={formData.confirm_password}
                onChange={(e) => setFormData({...formData, confirm_password: e.target.value})}
              />
            </div>
            <Button className="w-full" type="submit" disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Ouvrir mon compte virtuel
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex justify-center border-t p-4 text-sm">
          <span className="text-muted-foreground mr-1">Déjà membre ?</span>
          <Link href="/login" className="text-primary font-medium hover:underline">
            Connectez-vous
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
