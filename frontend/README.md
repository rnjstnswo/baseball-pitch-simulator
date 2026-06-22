# Frontend

The React + TypeScript frontend is scaffolded in Phase 5.

## Setup (Phase 5)

Run the following command from the **repo root** to scaffold the Vite app:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite recharts @tanstack/react-query
```

## Tech Stack

- **React + TypeScript** via Vite
- **Tailwind CSS** for styling
- **Recharts** for probability bar charts
- **TanStack Query** for API data fetching
- **Hand-rolled SVG** for the interactive strike zone

## API

The frontend expects the API to be running at `http://localhost:8000`.
Configure via the `VITE_API_URL` environment variable.
