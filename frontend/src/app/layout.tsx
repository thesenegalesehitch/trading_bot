import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Quantum Trading Platform",
  description: "Plateforme de référence mondiale pour le trading quantitatif et éducatif.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr" className="dark">
      <body className={`${inter.className} min-h-screen bg-background text-foreground antialiased`}>
        {children}
        <Toaster />
      </body>
    </html>
  );
}
