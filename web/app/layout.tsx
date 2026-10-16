import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Riftside Oracle | LoL Esports Predictions",
  description: "XGBoost-powered League of Legends esports match predictions with player ELO ratings and SHAP explanations.",
  openGraph: {
    title: "Riftside Oracle",
    description: "League of Legends esports match predictions powered by machine learning.",
    url: "https://riftside-oracle.gerritvisser.de",
    siteName: "Riftside Oracle",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-brand-bg antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
