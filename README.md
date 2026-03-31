# football_prediction

`football_prediction` ist ein Python-Paket zur Vorhersage von Fußballspielergebnissen auf Basis historischer Spieldaten.  
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

## Projektstruktur

```text
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
