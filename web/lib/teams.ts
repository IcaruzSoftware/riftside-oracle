export const TEAM_IDS: Record<string, number> = {
  ATL: 1610612737, BOS: 1610612738, BKN: 1610612751, CHA: 1610612766,
  CHI: 1610612741, CLE: 1610612739, DAL: 1610612742, DEN: 1610612743,
  DET: 1610612765, GSW: 1610612744, HOU: 1610612745, IND: 1610612754,
  LAC: 1610612746, LAL: 1610612747, MEM: 1610612763, MIA: 1610612748,
  MIL: 1610612749, MIN: 1610612750, NOP: 1610612740, NYK: 1610612752,
  OKC: 1610612760, ORL: 1610612753, PHI: 1610612755, PHX: 1610612756,
  POR: 1610612757, SAC: 1610612758, SAS: 1610612759, TOR: 1610612761,
  UTA: 1610612762, WAS: 1610612764,
};

export const TEAM_NAMES: Record<string, string> = {
  ATL: "Hawks",    BOS: "Celtics",   BKN: "Nets",      CHA: "Hornets",
  CHI: "Bulls",    CLE: "Cavaliers", DAL: "Mavericks", DEN: "Nuggets",
  DET: "Pistons",  GSW: "Warriors",  HOU: "Rockets",   IND: "Pacers",
  LAC: "Clippers", LAL: "Lakers",    MEM: "Grizzlies", MIA: "Heat",
  MIL: "Bucks",    MIN: "Timberwolves", NOP: "Pelicans", NYK: "Knicks",
  OKC: "Thunder",  ORL: "Magic",     PHI: "76ers",     PHX: "Suns",
  POR: "Trail Blazers", SAC: "Kings", SAS: "Spurs",    TOR: "Raptors",
  UTA: "Jazz",     WAS: "Wizards",
};

export function teamLogoUrl(abbrevOrId: string): string {
  const id = TEAM_IDS[abbrevOrId] ?? parseInt(abbrevOrId, 10);
  return `https://cdn.nba.com/logos/nba/${id}/global/L/logo.svg`;
}
