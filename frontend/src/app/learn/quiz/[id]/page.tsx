"use client";

import { useEffect, useState, use } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api';
import { ChevronLeft, Trophy, CheckCircle2, XCircle, HelpCircle } from 'lucide-react';
import Link from 'next/link';

export default function QuizPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [quiz, setQuiz] = useState<any>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [isAnswered, setIsAnswered] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchQuiz = async () => {
      try {
        const res = await apiClient.get(`/academy/courses/${id}/quiz`);
        setQuiz(res.data);
      } catch (error) {
        console.error("Failed to fetch quiz", error);
      } finally {
        setLoading(false);
      }
    };
    fetchQuiz();
  }, [id]);

  const handleOptionSelect = (optionId: number) => {
    if (isAnswered) return;
    setSelectedOption(optionId);
  };

  const handleConfirm = () => {
    if (selectedOption === null || isAnswered) return;
    
    const question = quiz.questions[currentQuestionIndex];
    const option = question.options.find((o: any) => o.id === selectedOption);
    
    if (option.is_correct) {
      setScore(score + 1);
    }
    
    setIsAnswered(true);
  };

  const handleNext = () => {
    if (currentQuestionIndex < quiz.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedOption(null);
      setIsAnswered(false);
    } else {
      setShowResult(true);
    }
  };

  if (loading) return <div className="flex items-center justify-center min-h-screen">Chargement du quizz...</div>;
  if (!quiz) return <div className="flex flex-col items-center justify-center min-h-screen gap-4">
    <p>Aucun quizz disponible pour ce cours.</p>
    <Link href="/learn"><Button>Retour</Button></Link>
  </div>;

  const currentQuestion = quiz.questions[currentQuestionIndex];

  if (showResult) {
    const finalScore = (score / quiz.questions.length) * 100;
    return (
      <div className="flex bg-muted/20 min-h-screen">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <Card className="w-full max-w-lg text-center p-8 shadow-2xl border-primary/20">
            <div className="mb-6 flex justify-center">
              <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
                <Trophy className="w-10 h-10 text-primary" />
              </div>
            </div>
            <CardTitle className="text-3xl mb-2">Quizz Terminé !</CardTitle>
            <p className="text-muted-foreground mb-8 text-lg">
              Votre score : <span className="text-primary font-bold">{finalScore.toFixed(0)}%</span>
            </p>
            <div className="space-y-4">
              <p className="text-sm">
                {finalScore >= 70 ? "Félicitations ! Vous avez maîtrisé les concepts de ce module." : "Nous vous conseillons de revoir les leçons pour améliorer vos connaissances."}
              </p>
              <Link href="/learn" className="block w-full">
                <Button className="w-full h-12 text-lg">Retourner à l'Académie</Button>
              </Link>
            </div>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="flex bg-muted/20 min-h-screen">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto">
        <Link href={`/learn/${id}`} className="flex items-center text-sm text-muted-foreground hover:text-primary mb-6 transition-colors">
          <ChevronLeft className="w-4 h-4 mr-1" /> Retour au cours
        </Link>

        <div className="max-w-3xl mx-auto">
          <div className="flex justify-between items-center mb-6 text-sm font-medium">
            <span className="text-primary">Question {currentQuestionIndex + 1} sur {quiz.questions.length}</span>
            <span className="bg-muted px-3 py-1 rounded-full">Score: {score}</span>
          </div>

          <Card className="shadow-xl border-primary/10 overflow-hidden">
            <div className="h-2 bg-muted w-full">
              <div 
                className="h-full bg-primary transition-all duration-500" 
                style={{ width: `${((currentQuestionIndex + 1) / quiz.questions.length) * 100}%` }}
              />
            </div>
            <CardHeader className="pt-8">
              <CardTitle className="text-xl leading-relaxed">{currentQuestion.text}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pt-4">
              {currentQuestion.options.map((option: any) => {
                let statusClass = "border-muted hover:border-primary/50 bg-card";
                if (isAnswered) {
                  if (option.is_correct) statusClass = "border-emerald-500 bg-emerald-500/10 text-emerald-600";
                  else if (selectedOption === option.id) statusClass = "border-red-500 bg-red-500/10 text-red-600";
                  else statusClass = "opacity-50 border-muted bg-muted/5";
                } else if (selectedOption === option.id) {
                  statusClass = "border-primary bg-primary/5 ring-1 ring-primary";
                }

                return (
                  <button
                    key={option.id}
                    onClick={() => handleOptionSelect(option.id)}
                    className={`w-full text-left p-4 rounded-xl border-2 transition-all flex items-center justify-between group ${statusClass}`}
                    disabled={isAnswered}
                  >
                    <span className="font-medium">{option.text}</span>
                    {isAnswered && option.is_correct && <CheckCircle2 className="w-5 h-5" />}
                    {isAnswered && selectedOption === option.id && !option.is_correct && <XCircle className="w-5 h-5" />}
                  </button>
                );
              })}

              {isAnswered && currentQuestion.explanation && (
                <div className="mt-6 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20 text-sm text-blue-400 flex gap-3 animate-in fade-in slide-in-from-top-2">
                  <HelpCircle className="w-5 h-5 flex-shrink-0" />
                  <p>{currentQuestion.explanation}</p>
                </div>
              )}
            </CardContent>
            <CardFooter className="bg-muted/5 border-t p-6 flex justify-end">
              {!isAnswered ? (
                <Button 
                  onClick={handleConfirm} 
                  disabled={selectedOption === null}
                  className="px-8 h-12 text-lg"
                >
                  Confirmer la réponse
                </Button>
              ) : (
                <Button 
                  onClick={handleNext} 
                  className="px-8 h-12 text-lg gap-2"
                >
                  {currentQuestionIndex < quiz.questions.length - 1 ? "Question suivante" : "Voir mon score"}
                  <ArrowRight className="w-5 h-5" />
                </Button>
              )}
            </CardFooter>
          </Card>
        </div>
      </main>
    </div>
  );
}
