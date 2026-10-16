import Image from "next/image";

interface Props {
  src: string | null;
  code: string;
  size?: number;
}

export default function TeamLogo({ src, code, size = 56 }: Props) {
  if (!src) {
    return (
      <div
        className="rounded-full bg-brand-border flex items-center justify-center text-[10px] font-bold text-brand-muted"
        style={{ width: size, height: size }}
      >
        {code}
      </div>
    );
  }
  return (
    <Image
      src={src.replace(/^http:\/\/static\.lolesports\.com\//, "https://static.lolesports.com/")}
      alt={code}
      width={size}
      height={size}
      className="object-contain drop-shadow-md"
      unoptimized
    />
  );
}
