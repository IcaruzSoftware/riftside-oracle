import Image from "next/image";
import { teamLogoUrl } from "@/lib/teams";

interface Props {
  abbrev: string;
  teamId?: string | null;
  size?: number;
}

export default function TeamLogo({ abbrev, teamId, size = 56 }: Props) {
  const src = teamId ? teamLogoUrl(teamId) : teamLogoUrl(abbrev);
  return (
    <Image
      src={src}
      alt={abbrev}
      width={size}
      height={size}
      className="object-contain drop-shadow-md"
      unoptimized
    />
  );
}
