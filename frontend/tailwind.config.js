/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pivai: {
          black: '#0a0a0b',
          black2: '#121214',
          black3: '#1b1b1e',
          ink: '#f3f1ec',
          inkDim: 'rgba(243, 241, 236, 0.66)',
          inkFaint: 'rgba(243, 241, 236, 0.42)',
          paper: '#f3f1ec',
          paper2: '#e9e6df',
          text: '#121214',
          textDim: '#55555b',
          lineDark: 'rgba(255, 255, 255, 0.12)',
          lineLight: 'rgba(18, 18, 20, 0.12)',
          amber: '#f5a524',
          amberHover: '#ffb840',
          amberDeep: '#d98a0b',
        },
        brand: {
          navy: '#101D30',
          sidebar: '#0a0a0b',
          bg: '#F5F7FA',
          card: '#FFFFFF',
          text: '#121214',
          muted: '#55555b',
          border: '#E3E8EF',
          accent: '#f5a524',
          accentHover: '#ffb840'
        },
        navy: {
          950: '#0a0a0b',
          900: '#101D30',
          800: '#14243A',
          700: '#1E324F',
          600: '#2A4365',
          500: '#3B82F6'
        },
        pothole: {
          50: '#FEF2F2',
          100: '#FEE2E2',
          500: '#FF3B30',
          600: '#E02D23',
          700: '#B91C1C',
          DEFAULT: '#FF3B30'
        },
        crack: {
          50: '#FFFBEB',
          100: '#FEF3C7',
          500: '#FFCC00',
          600: '#D97706',
          700: '#B45309',
          DEFAULT: '#FFCC00'
        }
      },
      fontFamily: {
        sans: ['"Inter Tight"', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'xs': '0 1px 2px 0 rgba(16, 24, 40, 0.05)',
        'subtle': '0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02)',
        'card': '0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.04)',
        'elevated': '0 10px 25px -5px rgba(10, 10, 11, 0.12), 0 8px 10px -6px rgba(10, 10, 11, 0.08)'
      }
    },
  },
  plugins: [],
}
