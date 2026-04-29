"use client";

import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { User, Bell, Shield, Wallet } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <h1 className="text-3xl font-bold mb-8">Paramètres</h1>

        <div className="max-w-4xl space-y-8">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <User className="w-5 h-5 text-primary" />
                <CardTitle>Profil Utilisateur</CardTitle>
              </div>
              <CardDescription>Gérez vos informations personnelles et votre identité.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Nom Complet</Label>
                  <Input placeholder="Alexandre Ndour" />
                </div>
                <div className="space-y-2">
                  <Label>Email</Label>
                  <Input type="email" placeholder="alex@quantum.com" />
                </div>
              </div>
              <Button>Sauvegarder les modifications</Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-emerald-500" />
                <CardTitle>Sécurité & API</CardTitle>
              </div>
              <CardDescription>Configurez vos clés API pour le trading Live (Optionnel).</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Clé API Broker (Binance/IG)</Label>
                <Input type="password" placeholder="••••••••••••••••" />
              </div>
              <Button variant="outline">Connecter un Broker</Button>
            </CardContent>
          </Card>

          <Card className="border-destructive/20">
            <CardHeader>
              <CardTitle className="text-destructive">Zone de Danger</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">La suppression de votre compte est irréversible.</p>
              <Button variant="destructive">Supprimer mon compte Quantum</Button>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
