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
        // Archivo's unicode-range covers Latin + figures only, so any Chinese
        // inside a data element still resolves to the same face as body text.
        data: ['Archivo', 'PingFang TC', 'Noto Sans TC', 'system-ui', 'sans-serif']
      },
      fontSize: {
        // One deliberate jump from body to score. Nothing lives in between.
        score: ['3.75rem', { lineHeight: '0.88', letterSpacing: '-0.035em', fontWeight: '700' }],
        metric: ['1.375rem', { lineHeight: '1.1', letterSpacing: '-0.02em', fontWeight: '600' }]
      },
      borderRadius: {
        '2xl': 'calc(var(--radius) * 2)',
        xl: 'calc(var(--radius) + 2px)',
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 1px)',
        sm: 'calc(var(--radius) - 2px)'
      },
      boxShadow: {
        // Barely-there. Structure comes from rules and colour, not elevation.
        card: '0 1px 2px hsl(193 30% 12% / 0.05)',
        elevated: '0 8px 28px -12px hsl(193 30% 12% / 0.28)'
      },
      maxWidth: {
        content: '46rem'
      }
    }
  },
  plugins: [require('tailwindcss-animated'), require('tailwindcss-animate')]
} satisfies Config
