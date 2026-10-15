import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Courtside Oracle | NBA Predictions",
  description: "XGBoost-powered NBA game outcome predictions with player ELO ratings and SHAP explanations.",
  openGraph: {
    title: "Courtside Oracle",
    description: "NBA game predictions powered by machine learning.",
    url: "https://courtside-oracle.gerritvisser.de",
    siteName: "Courtside Oracle",
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
