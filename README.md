# Aktien-Qualitätsscreener

Website, die rund 500 der größten börsennotierten Unternehmen aus Nordamerika, Europa und Asien-Pazifik mehrmals täglich nach festen Regeln bewertet. Mit Chartanalyse, Analystenbild, Nachrichten und einer Depotauswertung im Browser.
Keine Anlageberatung, keine Kauf- oder Verkaufsempfehlung.

## Aufbau

| Datei | Aufgabe |
| --- | --- |
| `docs/index.html` | Die Website. Lädt `docs/data.json`. |
| `scripts/build_universe.py` | Bestimmt das Universum nach Regionen-Kontingenten (monatlich). |
| `scripts/update.py` | Lädt Kurse und Fundamentaldaten von Yahoo Finance, schreibt `docs/data.json`, `docs/charts.json`, `docs/history.json`. |
| `scripts/scoring.py` | Die Bewertungsregeln, alle Schwellen und die Datenprüfungen. |
| `scripts/selftest.py` | Prüft die Rechenregeln an festen Beispielen. Läuft vor jedem Datenlauf. |
| `.github/workflows/update.yml` | Startet den Lauf dreimal täglich automatisch. |

## Einrichtung

1. Repository öffentlich anlegen und diese Dateien hochladen.
2. Settings → Pages → Source „Deploy from a branch“, Branch `main`, Ordner `/docs`.
3. Settings → Actions → General → Workflow permissions: „Read and write permissions“.
4. Actions → „Daten aktualisieren“ → „Run workflow“. Der erste Lauf dauert ein bis zwei Stunden.

## Regeln ändern

Schwellen stehen oben in `scripts/scoring.py` (`TH`). Nach einer Änderung rechnet der nächste Lauf alles neu.
