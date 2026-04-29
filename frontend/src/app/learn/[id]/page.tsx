"use client";

import { useEffect, useState, use } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChevronLeft, CheckCircle, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export default function CourseDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [lessons, setLessons] = useState<any[]>([]);
  const [selectedLesson, setSelectedLesson] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLessons = async () => {
      try {
        const res = await apiClient.get(`/academy/courses/${id}/lessons`);
        setLessons(res.data);
        if (res.data.length > 0) setSelectedLesson(res.data[0]);
      } catch (error) {
        console.error("Failed to fetch lessons", error);
      } finally {
        setLoading(false);
      }
    };
    fetchLessons();
  }, [id]);

  if (loading) return <div className="flex items-center justify-center min-h-screen">Chargement...</div>;

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <Link href="/learn" className="flex items-center text-sm text-muted-foreground hover:text-primary mb-6 transition-colors">
          <ChevronLeft className="w-4 h-4 mr-1" /> Retour à l'académie
        </Link>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Lesson List */}
          <div className="lg:col-span-1 space-y-4">
            <h2 className="font-bold text-lg px-2">Sommaire</h2>
            {lessons.map((lesson) => (
              <button
                key={lesson.id}
                onClick={() => setSelectedLesson(lesson)}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  selectedLesson?.id === lesson.id 
                    ? 'bg-primary/10 border-primary text-primary font-medium shadow-sm' 
                    : 'bg-card hover:bg-muted text-muted-foreground'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-xs opacity-50">{lesson.order}.</span>
                  <span className="truncate">{lesson.title}</span>
                </div>
              </button>
            ))}
            <Link href={`/learn/quiz/${id}`} className="block">
              <Button variant="outline" className="w-full border-dashed border-primary/50 text-primary hover:bg-primary/5">
                Passer le Quizz final
              </Button>
            </Link>
          </div>

          {/* Lesson Content */}
          <div className="lg:col-span-3">
            {selectedLesson ? (
              <Card className="shadow-lg border-primary/10">
                <CardHeader className="border-b bg-muted/5">
                  <CardTitle className="text-2xl">{selectedLesson.title}</CardTitle>
                </CardHeader>
                <CardContent className="pt-8 prose prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {selectedLesson.content}
                  </ReactMarkdown>
                  
                  <div className="mt-12 flex justify-between items-center pt-8 border-t">
                    <div className="text-sm text-muted-foreground">
                      Temps estimé : {selectedLesson.duration}
                    </div>
                    <Button className="gap-2">
                      Marquer comme complété <CheckCircle className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="text-center py-20 text-muted-foreground">
                Sélectionnez une leçon pour commencer.
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
