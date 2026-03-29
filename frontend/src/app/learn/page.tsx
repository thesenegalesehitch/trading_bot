"use client";

import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { PlayCircle, Target, Trophy, Clock, Lock } from 'lucide-react';

export default function LearnPage() {
  const [courses] = useState([
    {
      id: 1,
      title: "Indépendance et Discipline",
      description: "Apprenez les bases de la psychologie du trading et la rigueur institutionnelle.",
      progress: 100,
      duration: "45 min",
      locked: false,
    },
    {
      id: 2,
      title: "Gestion du Risque & VaR",
      description: "Utilisation pratique du Calculator VaR, Kelly Criterion et Stress Testing.",
      progress: 60,
      duration: "1h 20m",
      locked: false,
    },
    {
      id: 3,
      title: "Méthode Wyckoff (Fondations)",
      description: "Identifier Accumulation et Distribution avant les cassures retail.",
      progress: 0,
      duration: "2h 15m",
      locked: false,
    },
    {
      id: 4,
      title: "Smart Money Concepts (SMC)",
      description: "Order Blocks, Fair Value Gaps et Inefficiencies de marché expliqués.",
      progress: 0,
      duration: "3h 00m",
      locked: true,
    },
    {
      id: 5,
      title: "Inner Circle Trader (ICT) Avancé",
      description: "Killzones, Liquidity Sweeps et Market Structure Shifts de haute précision.",
      progress: 0,
      duration: "4h 30m",
      locked: true,
    }
  ]);

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Académie Quantum</h1>
          <p className="text-muted-foreground">Formez-vous aux méthodes institutionnelles de A à Z.</p>
        </div>

        {/* Global Progress */}
        <Card className="mb-8 bg-primary/5 border-primary/20">
          <CardContent className="pt-6 flex flex-col md:flex-row items-center gap-6">
            <div className="p-4 bg-primary/10 rounded-full">
              <Trophy className="w-12 h-12 text-primary" />
            </div>
            <div className="flex-1 space-y-2">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">Votre Progression Globale</h2>
                <span className="font-bold text-primary">32%</span>
              </div>
              <Progress value={32} className="h-2" />
              <p className="text-sm text-muted-foreground">Continuez ainsi ! Prochaine étape : Compléter le module de Gestion du Risque.</p>
            </div>
          </CardContent>
        </Card>

        {/* Course List */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {courses.map((course) => (
            <Card key={course.id} className={`transition-all ${course.locked ? 'opacity-70 bg-muted/50' : 'hover:border-primary/50 hover:shadow-md'}`}>
              <CardHeader className="pb-4">
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-lg flex items-center gap-2">
                      {course.locked && <Lock className="w-4 h-4 text-muted-foreground" />}
                      M{course.id} : {course.title}
                    </CardTitle>
                    <CardDescription className="mt-2 line-clamp-2">{course.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> {course.duration}</span>
                  <span className="flex items-center gap-1"><Target className="w-4 h-4" /> {course.progress}% complété</span>
                </div>
                {!course.locked && <Progress value={course.progress} className="h-1.5" />}
              </CardContent>
              <CardFooter>
                <Button 
                  className="w-full" 
                  variant={course.progress === 100 ? "outline" : course.locked ? "secondary" : "default"}
                  disabled={course.locked}
                >
                  {course.progress === 100 ? "Revoir le cours" : 
                   course.progress > 0 ? "Continuer" : 
                   course.locked ? "Ouvrez le module précédent" : "Commencer"}
                  {!course.locked && course.progress !== 100 && <PlayCircle className="w-4 h-4 ml-2" />}
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </main>
    </div>
  );
}
