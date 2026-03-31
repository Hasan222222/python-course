# football_prediction

`football_prediction` ist ein Python-Paket zur Vorhersage von Fußballspielergebnissen auf Basis historischer Spieldaten aus der Primer League (von 2020-2026).  
Das Projekt lädt Rohdaten aus CSV-Dateien, bereinigt und transformiert diese Daten, erzeugt Merkmale zur Teamform und 
trainiert anschließend Poisson-Modelle, um erwartete Tore sowie Wahrscheinlichkeiten für Heimsieg, Unentschieden 
und Auswärtssieg zu berechnen.

Zusätzlich erstellt das Paket Visualisierungen der Ergebniswahrscheinlichkeiten und speichert diese als PNG-Dateien.

---

## Projektziel

Ziel dieses Projekts ist es, ein installierbares und ausführbares Python-Paket zu entwickeln, das Fußballspiele 
modelliert und Vorhersagen für konkrete Begegnungen trifft.

Für ein gegebenes Spiel zwischen Heim- und Auswärtsteam werden berechnet:

- erwartete Tore des Heimteams
- erwartete Tore des Auswärtsteams
- gerundetes prognostiziertes Endergebnis
- Sieger-Tipp
- Wahrscheinlichkeit für Heimsieg
- Wahrscheinlichkeit für Unentschieden
- Wahrscheinlichkeit für Auswärtssieg

Außerdem werden Visualisierungen der Vorhersage erzeugt.

---

## Verwendete Methoden

Das Projekt basiert auf einem modellbasierten Ansatz in mehreren Schritten:

1. **Einlesen der Rohdaten**
   - CSV-Dateien werden aus dem Ordner `data/raw` geladen und zu einem gemeinsamen Datensatz zusammengeführt.

2. **Datenbereinigung**
   - Relevante Variablen werden ausgewählt und umbenannt.
   - Verwendet werden insbesondere:
     - Datum
     - Heimteam
     - Auswärtsteam
     - Heimtore
     - Auswärtstore
     - Endergebnis

3. **Feature Engineering**
   - Erstellung teambezogener Verlaufsmerkmale
   - Rolling averages der letzten Spiele
   - Merkmale für:
     - erzielte Tore
     - kassierte Tore
     - Punkte
     - Tordifferenz
   - getrennte Betrachtung von Heim- und Auswärtsform

4. **Modellierung**
   - Training von zwei Poisson-Regressionsmodellen:
     - ein Modell für Heimtore
     - ein Modell für Auswärtstore

5. **Vorhersage**
   - Berechnung erwarteter Tore für beide Teams
   - Berechnung von Ergebniswahrscheinlichkeiten mit der Poisson-Verteilung

6. **Visualisierung**
   - Balkendiagramm für Sieg / Unentschieden / Niederlage
   - Heatmap exakter Ergebniswahrscheinlichkeiten
   - Verteilung der Heimtore
   - Verteilung der Auswärtstore

---

## Wissenschaftliche Fragestellungen

Im Rahmen des Projekts wurden zusätzlich zwei wissenschaftliche Fragestellungen untersucht.

### 1. Sind Tore des Heimteams besser vorhersagbar als Auswärtstore?

Zur Beantwortung dieser Frage wurde die Vorhersagegüte der beiden separat trainierten Poisson-Modelle verglichen.  
Dazu wurden die tatsächlichen Tore mit den vorhergesagten Toren für Heim- und Auswärtsteams getrennt betrachtet.  
Als Gütemaße wurden der Mean Absolute Error (MAE) und die Root Mean Squared Error (RMSE) verwendet.

**Ergebnisse:**
- MAE Heimtore: **1.043**
- RMSE Heimtore: **1.308**
- MAE Auswärtstore: **0.965**
- RMSE Auswärtstore: **1.212**

**Interpretation:**  
Die Analyse liefert keine Evidenz dafür, dass Heimtore besser vorhersagbar sind als Auswärtstore.  
Im verwendeten Modell weisen Auswärtstore sowohl beim MAE als auch beim RMSE geringere Fehler auf.  
Damit sind Auswärtstore in diesem Datensatz geringfügig besser vorhersagbar als Heimtore.

---

### 2. Wie gut passt die Poisson-Verteilung zu den tatsächlichen Toren?

Zur Beantwortung dieser Frage wurde die beobachtete Verteilung der erzielten Tore mit einer theoretischen Poisson-Verteilung verglichen.  
Dazu wurden für Heim- und Auswärtstore jeweils die empirischen Häufigkeiten den entsprechenden Poisson-Wahrscheinlichkeiten gegenübergestellt.  
Zusätzlich wurden Mittelwert und Varianz der Torverteilungen verglichen, da bei einer idealen Poisson-Verteilung beide Größen ungefähr übereinstimmen.

**Ergebnisse:**
- Heimtore:
  - Mittelwert: **1.563**
  - Varianz: **1.822**
- Auswärtstore:
  - Mittelwert: **1.353**
  - Varianz: **1.528**

**Interpretation:**  
Die grafischen Vergleiche zeigen insgesamt eine gute Übereinstimmung zwischen beobachteter Torverteilung und theoretischer Poisson-Verteilung.  
Sowohl für Heim- als auch für Auswärtstore folgt die beobachtete Verteilung dem typischen Verlauf der Poisson-Verteilung relativ nah.  
Die Varianz liegt jedoch jeweils etwas über dem Mittelwert, was auf eine leichte Überdispersion hinweist.  
Die Poisson-Annahme ist damit nicht perfekt erfüllt, stellt für die Modellierung von Fußballtoren in diesem Datensatz jedoch eine sinnvolle und gut passende Grundlage dar.

---

## Tests

Zur Überprüfung der Funktionsfähigkeit des Pakets wurden automatische Tests mit `pytest` implementiert.  
Dabei wurden insbesondere zentrale mathematische und modellbezogene Funktionen geprüft.

Getestet wurden unter anderem:

- die korrekte Berechnung der Poisson-Wahrscheinlichkeitsfunktion (`poisson_pmf`)
- die Berechnung von Ergebniswahrscheinlichkeiten
- die Struktur und Inhalte der Vorhersageausgaben
- die Funktionen `predict_match_full` und `predict_match_full2`

Die Tests befinden sich im Ordner `tests/` und können mit folgendem Befehl ausgeführt werden:

```bash
pytest

## Projektstrucktur 
football-prediction/
├── data/
│   └── raw/
│       └── *.csv
├── outputs/
│   └── (automatisch erzeugte Grafiken)
├── src/
│   └── football_prediction/
│       ├── __main__.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── features.py
│       └── model.py
├── tests/
├── Notebooks/
├── pyproject.toml
└── README.md

