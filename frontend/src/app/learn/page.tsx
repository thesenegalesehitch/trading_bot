"use client";

import { useEffect, useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { PlayCircle, Target, Trophy, Clock, Lock } from 'lucide-react';
import { apiClient } from '@/lib/api';
import Link from 'next/link';

export default function LearnPage() {
  const [courses, setCourses] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCourses = async () => {
      try {
        const res = await apiClient.get('/academy/courses');
        setCourses(res.data);
      } catch (error) {
        console.error("Failed to fetch courses", error);
      } finally {
        setLoading(false);
      }
    };
    fetchCourses();
  }, []);

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
                <span className="font-bold text-primary">0%</span>
              </div>
              <Progress value={0} className="h-2" />
              <p className="text-sm text-muted-foreground">Commencez votre premier module pour débloquer votre potentiel.</p>
            </div>
          </CardContent>
        </Card>

        {loading ? (
          <div className="text-center py-12">Chargement des cours...</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {courses.map((course, idx) => (
              <Card key={course.id} className="hover:border-primary/50 hover:shadow-md transition-all">
                <CardHeader className="pb-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-lg">
                        Module {idx + 1} : {course.title}
                      </CardTitle>
                      <CardDescription className="mt-2 line-clamp-2">{course.description}</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> {course.level}</span>
                    <span className="flex items-center gap-1"><Target className="w-4 h-4" /> 0% complété</span>
                  </div>
                  <Progress value={0} className="h-1.5" />
                </CardContent>
                <CardFooter>
                  <Link href={`/learn/${course.id}`} className="w-full">
                    <Button className="w-full">
                      Commencer <PlayCircle className="w-4 h-4 ml-2" />
                    </Button>
                  </Link>
                </CardFooter>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
