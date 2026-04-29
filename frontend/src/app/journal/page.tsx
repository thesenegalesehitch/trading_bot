"use client";

import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BookOpen, Plus, Smile } from 'lucide-react';

export default function JournalPage() {
  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold">Journal de Trading</h1>
            <Button>
                <Plus className="w-4 h-4 mr-2" />
                Nouvelle Note
            </Button>
        </div>

        <div className="grid grid-cols-1 gap-6">
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Smile className="w-5 h-5 text-emerald-500" />
                        Session Positive - Discipline Respectée
                    </CardTitle>
                    <CardDescription>29 Avril 2026 • Stratégie ICT</CardDescription>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                        Aujourd'hui j'ai attendu le sweep de liquidité sur le BTC avant d'entrer en position. 
                        Mon ratio risque/récompense était de 1:3. J'ai gardé mon calme malgré une légère mèche contre moi.
                    </p>
                </CardContent>
            </Card>

            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground border-2 border-dashed rounded-xl">
                <BookOpen className="w-16 h-16 mb-4 opacity-10" />
                <p>Traquez vos émotions pour devenir un trader discipliné.</p>
            </div>
        </div>
      </main>
    </div>
  );
}
