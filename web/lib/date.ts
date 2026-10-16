/**
 * Returns a date formatted as "YYYY-MM-DD" in the viewer's local time zone.
 * Matches are grouped by the viewer's local date of `start_time_utc`, not by the
 * `match_date` column (which is the UTC date).
 */
export function getLocalDateString(date: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

/** Start (inclusive) and end (exclusive) of the viewer's local day containing `date`. */
export function getLocalDayBounds(date: Date): [Date, Date] {
  const [y, m, d] = [date.getFullYear(), date.getMonth(), date.getDate()];
  return [new Date(y, m, d), new Date(y, m, d + 1)];
}
