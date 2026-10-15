/**
 * Returns a date formatted as "YYYY-MM-DD" in US Eastern time (America/New_York),
 * matching the format of the `game_date` column. Using UTC (e.g. `toISOString()`)
 * is wrong here because it rolls over to the next day ~8pm ET.
 */
export function getEasternDateString(date: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}
