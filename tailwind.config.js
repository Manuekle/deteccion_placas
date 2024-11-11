/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html", // Incluye todos los archivos HTML en la carpeta 'templates'
    "./**/*.html", // Incluye cualquier otro archivo HTML en la raíz o subdirectorios
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
