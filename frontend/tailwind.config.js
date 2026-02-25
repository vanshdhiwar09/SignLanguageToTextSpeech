/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'cyber-dark': '#0a0e27',
                'cyber-darker': '#050814',
                'cyber-blue': '#00d9ff',
                'cyber-purple': '#b537f2',
                'cyber-pink': '#ff2e97',
                'cyber-green': '#00ff88',
            },
            fontFamily: {
                'sans': ['Inter', 'system-ui', 'sans-serif'],
                'mono': ['JetBrains Mono', 'monospace'],
            },
            boxShadow: {
                'neon-blue': '0 0 20px rgba(0, 217, 255, 0.5)',
                'neon-purple': '0 0 20px rgba(181, 55, 242, 0.5)',
                'neon-green': '0 0 20px rgba(0, 255, 136, 0.5)',
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glow': 'glow 2s ease-in-out infinite alternate',
            },
            keyframes: {
                glow: {
                    '0%': { boxShadow: '0 0 5px rgba(0, 217, 255, 0.5)' },
                    '100%': { boxShadow: '0 0 20px rgba(0, 217, 255, 0.8)' },
                }
            }
        },
    },
    plugins: [],
}
