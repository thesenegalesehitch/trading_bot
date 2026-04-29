import Link from 'next/link';
import { LayoutDashboard, BookOpen, Settings, LogOut, Zap, TrendingUp, ShieldCheck, History, GraduationCap, CandlestickChart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePathname, useRouter } from 'next/navigation';

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = () => {
    localStorage.removeItem('token');
    router.push('/login');
  };

  const navItems = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Trading Démo', href: '/trade', icon: TrendingUp },
    { name: 'Analyse', href: '/analysis', icon: Zap },
    { name: 'Gestion Risques', href: '/risk', icon: ShieldCheck },
    { name: 'Backtesting', href: '/backtest', icon: History },
    { name: 'Journal', href: '/journal', icon: BookOpen },
    { name: 'Académie', href: '/learn', icon: GraduationCap },
    { name: 'Paramètres', href: '/settings', icon: Settings },
  ];

  return (
    <div className="flex flex-col h-full w-64 bg-card border-r">
      <div className="p-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <CandlestickChart className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-bold tracking-tight">Quantum</span>
        </Link>
      </div>

      <nav className="flex-1 px-4 space-y-2 mt-4">
        {navItems.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors ${
                isActive 
                  ? 'bg-primary/20 text-primary font-medium' 
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              }`}
            >
              <item.icon className={`w-5 h-5 ${isActive ? 'text-primary' : ''}`} />
              {item.name}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t">
        <Button variant="ghost" className="w-full justify-start text-muted-foreground hover:text-red-400 hover:bg-red-400/10" onClick={handleLogout}>
          <LogOut className="w-4 h-4 mr-2" />
          Déconnexion
        </Button>
      </div>
    </div>
  );
}
