"use client";

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Calendar, Play, Pause, FastForward, Rewind } from 'lucide-react';

interface MarketReplayProps {
    onTimeChange: (timestamp: string | null) => void;
}

export function MarketReplay({ onTimeChange }: MarketReplayProps) {
    const [isActive, setIsActive] = useState(false);
    const [selectedTime, setSelectedTime] = useState("");

    const toggleReplay = () => {
        if (isActive) {
            onTimeChange(null);
            setIsActive(false);
        } else {
            setIsActive(true);
        }
    };

    const handleApply = () => {
        if (selectedTime) {
            onTimeChange(new Date(selectedTime).toISOString());
        }
    };

    return (
        <div className={`flex items-center gap-4 p-4 rounded-xl border transition-all ${isActive ? 'bg-primary/5 border-primary shadow-lg ring-1 ring-primary/20' : 'bg-card border-border'}`}>
            <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${isActive ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}>
                    <Calendar className="w-5 h-5" />
                </div>
                <div>
                    <p className="text-xs font-bold uppercase tracking-wider opacity-60">Market Replay</p>
                    <p className="text-sm font-semibold">{isActive ? "Mode Simulation Actif" : "Mode Temps Réel"}</p>
                </div>
            </div>

            <div className="flex-1 flex items-center gap-2">
                <Input 
                    type="datetime-local" 
                    value={selectedTime}
                    onChange={(e) => setSelectedTime(e.target.value)}
                    disabled={!isActive}
                    className="h-9 bg-background/50 border-muted"
                />
                <Button 
                    size="sm" 
                    variant={isActive ? "default" : "outline"}
                    onClick={handleApply}
                    disabled={!isActive || !selectedTime}
                >
                    Appliquer
                </Button>
            </div>

            <div className="flex items-center gap-1 bg-muted/30 p-1 rounded-lg border">
                <Button variant="ghost" size="icon" className="h-8 w-8" disabled={!isActive}><Rewind className="w-4 h-4" /></Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggleReplay}>
                    {isActive ? <Pause className="w-4 h-4 text-primary" /> : <Play className="w-4 h-4" />}
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8" disabled={!isActive}><FastForward className="w-4 h-4" /></Button>
            </div>
        </div>
    );
}
