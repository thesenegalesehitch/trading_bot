"use client";

import { useState, useEffect } from 'react';

export type ExperienceLevel = 'beginner' | 'intermediate' | 'expert';

export function useExperienceLevel() {
  const [level, setLevel] = useState<ExperienceLevel>('beginner');

  useEffect(() => {
    const saved = localStorage.getItem('quantum_exp_level') as ExperienceLevel;
    if (saved) setLevel(saved);
  }, []);

  const updateLevel = (newLevel: ExperienceLevel) => {
    setLevel(newLevel);
    localStorage.setItem('quantum_exp_level', newLevel);
  };

  const translate = (term: string, definition: string) => {
    if (level === 'beginner') {
      return definition;
    }
    return term;
  };

  return { level, updateLevel, translate };
}
