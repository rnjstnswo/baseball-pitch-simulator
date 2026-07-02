# Frontend

The React + TypeScript frontend (Phase 5).

## Setup

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (expects the API on :8000)
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
