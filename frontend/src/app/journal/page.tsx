"use client";

import { useEffect, useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { BookOpen, Plus, Smile, Meh, Frown, Loader2, Calendar } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { toast } from 'sonner';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';

export default function JournalPage() {
  const [entries, setEntries] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [isAdding, setIsAdding] = useState(false);
  
  // Form state
  const [mood, setMood] = useState('Satisfait');
  const [strategy, setStrategy] = useState('');
  const [notes, setNotes] = useState('');

  const fetchJournal = async () => {
    try {
      const res = await apiClient.get('/journal/');
      setEntries(res.data);
    } catch (error) {
      toast.error("Impossible de charger le journal");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJournal();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await apiClient.post('/journal/', {
        mood,
        strategy_name: strategy,
        notes: notes
      });
      toast.success("Note ajoutée au journal");
      setStrategy('');
      setNotes('');
      setIsAdding(false);
      fetchJournal();
    } catch (error) {
      toast.error("Erreur lors de l'ajout");
    }
  };

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="flex justify-between items-center mb-8">
            <div>
                <h1 className="text-3xl font-bold">Journal de Trading</h1>
                <p className="text-muted-foreground text-sm">Traquez vos émotions pour devenir un trader discipliné.</p>
            </div>
            <Button onClick={() => setIsAdding(!isAdding)} variant={isAdding ? "outline" : "default"}>
                {isAdding ? "Annuler" : <><Plus className="w-4 h-4 mr-2" /> Nouvelle Note</>}
            </Button>
        </div>

        {isAdding && (
            <Card className="mb-8 border-primary/20 animate-in fade-in slide-in-from-top-4">
                <CardHeader>
                    <CardTitle>Nouvelle Entrée</CardTitle>
                    <CardDescription>Décrivez votre session et votre état d'esprit.</CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label>Humeur</Label>
                                <div className="flex gap-2">
                                    {['Satisfait', 'Neutre', 'Stressé'].map((m) => (
                                        <Button 
                                            key={m}
                                            type="button" 
                                            variant={mood === m ? "default" : "outline"}
                                            size="sm"
                                            onClick={() => setMood(m)}
                                            className="flex-1"
                                        >
                                            {m === 'Satisfait' && <Smile className="w-4 h-4 mr-1" />}
                                            {m === 'Neutre' && <Meh className="w-4 h-4 mr-1" />}
                                            {m === 'Stressé' && <Frown className="w-4 h-4 mr-1" />}
                                            {m}
                                        </Button>
                                    ))}
                                </div>
                            </div>
                            <div className="space-y-2">
                                <Label>Stratégie utilisée</Label>
                                <Input value={strategy} onChange={e => setStrategy(e.target.value)} placeholder="ex: ICT Silver Bullet, Wyckoff accumulation..." />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <Label>Notes et leçons apprises</Label>
                            <Textarea 
                                value={notes} 
                                onChange={e => setNotes(e.target.value)} 
                                placeholder="Qu'avez-vous appris aujourd'hui ?"
                                className="min-h-[100px]"
                            />
                        </div>
                        <Button type="submit" className="w-full">Enregistrer dans le journal</Button>
                    </form>
                </CardContent>
            </Card>
        )}

        <div className="grid grid-cols-1 gap-6">
            {loading ? (
                <div className="flex justify-center py-12"><Loader2 className="animate-spin" /></div>
            ) : entries.length > 0 ? (
                entries.map((entry) => (
                    <Card key={entry.id} className="hover:shadow-md transition-shadow">
                        <CardHeader className="pb-2">
                            <div className="flex justify-between items-start">
                                <div className="flex items-center gap-2">
                                    {entry.mood === 'Satisfait' && <Smile className="w-5 h-5 text-emerald-500" />}
                                    {entry.mood === 'Neutre' && <Meh className="w-5 h-5 text-amber-500" />}
                                    {entry.mood === 'Stressé' && <Frown className="w-5 h-5 text-red-500" />}
                                    <CardTitle className="text-lg">{entry.mood}</CardTitle>
                                </div>
                                <div className="flex items-center text-xs text-muted-foreground gap-1">
                                    <Calendar className="w-3 h-3" />
                                    {format(new Date(entry.timestamp), 'PPP', { locale: fr })}
                                </div>
                            </div>
                            <CardDescription>Stratégie : <span className="text-foreground font-medium">{entry.strategy || 'N/A'}</span></CardDescription>
                        </CardHeader>
                        <CardContent>
                            <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                                {entry.notes}
                            </p>
                        </CardContent>
                    </Card>
                ))
            ) : (
                <div className="flex flex-col items-center justify-center py-20 text-muted-foreground border-2 border-dashed rounded-xl bg-muted/5">
                    <BookOpen className="w-16 h-16 mb-4 opacity-10" />
                    <p className="font-medium">Votre journal est vide.</p>
                    <p className="text-xs mt-2">Commencez par noter votre première session de trading.</p>
                </div>
            )}
        </div>
      </main>
    </div>
  );
}
