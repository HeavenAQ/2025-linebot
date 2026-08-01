import type { Config } from 'tailwindcss'

export default {
  darkMode: ['class'],
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}'
  ],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))'
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))'
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))'
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))'
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))'
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))'
        },
        highlight: {
          DEFAULT: 'hsl(var(--highlight))',
          foreground: 'hsl(var(--highlight-foreground))'
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))'
        },
        success: {
          DEFAULT: 'hsl(var(--success))',
          foreground: 'hsl(var(--success-foreground))'
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        chart: {
          '1': 'hsl(var(--chart-1))',
          '2': 'hsl(var(--chart-2))',
          '3': 'hsl(var(--chart-3))',
          '4': 'hsl(var(--chart-4))',
          '5': 'hsl(var(--chart-5))'
        }
      },
      fontFamily: {
        data: ['Archivo', 'PingFang TC', 'Noto Sans TC', 'system-ui', 'sans-serif']
      },
      fontSize: {
        // Large figures are set light and wide, not heavy — restraint at scale.
        figure: ['3rem', { lineHeight: '1', letterSpacing: '-0.02em', fontWeight: '400' }]
      },
      borderRadius: {
        '2xl': 'calc(var(--radius) * 3)',
        xl: 'calc(var(--radius) * 2)',
        lg: 'var(--radius)',
        md: 'var(--radius)',
        sm: 'calc(var(--radius) - 1px)'
      },
      spacing: {
        // 間 — the deliberate gaps between sections.
        ma: '3.5rem',
        'ma-sm': '2rem'
      },
      boxShadow: {
        // Depth is not part of this language. Kept only so old utilities resolve.
        card: 'none',
        elevated: '0 12px 40px -20px hsl(33 20% 12% / 0.35)'
      },
      maxWidth: {
        content: '44rem'
      }
    }
  },
  plugins: [require('tailwindcss-animated'), require('tailwindcss-animate')]
} satisfies Config
