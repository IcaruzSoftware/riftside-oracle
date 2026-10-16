import type { NextConfig } from "next";

if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
  console.warn(
    "⚠ NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY are not set (see web/.env.local.example): " +
      "the site will build but show no data. Preview with mock rows via /demo, or /?demo=1 and /card?demo=1 under `npm run dev`."
  );
}

const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
