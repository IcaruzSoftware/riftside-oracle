export default function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <nav className="border-b border-brand-border sticky top-0 z-50 bg-brand-bg/90 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center">
          <div className="flex items-center gap-2.5">
            <span className="text-brand-green text-xl">🔮</span>
            <span className="font-black tracking-tight text-white">
              RIFTSIDE <span className="text-brand-green">ORACLE</span>
            </span>
          </div>
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8">
        {children}
      </main>

      <footer className="border-t border-brand-border mt-16">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 space-y-3 text-xs text-brand-muted">
          <div className="flex items-center justify-between">
            <span>Riftside Oracle | XGBoost + Player ELO</span>
            <a
              href="https://gerritvisser.de"
              className="hover:text-white transition-colors"
            >
              gerritvisser.de
            </a>
          </div>
          <p>Match data: Oracle&apos;s Elixir and lolesports</p>
          <p>
            Riftside Oracle isn&apos;t endorsed by Riot Games and doesn&apos;t reflect the views or
            opinions of Riot Games or anyone officially involved in producing or managing Riot Games
            properties. Riot Games, and all associated properties are trademarks or registered
            trademarks of Riot Games, Inc.
          </p>
        </div>
      </footer>
    </>
  );
}
