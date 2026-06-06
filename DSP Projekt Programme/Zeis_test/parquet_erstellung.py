import pandas as pd
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import re
import gc

# ---------------------------------------------------------
# 1. Funktion: Rohdaten einlesen
# ---------------------------------------------------------
def erstelle_parquet():
    
    if Path("linsen_daten_roh.parquet").exists():
        print("Übersprungen: 'linsen_daten_roh.parquet' existiert bereits. Keine Ordner-Auswahl nötig.")
        return
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("Bitte wähle den Hauptordner mit den Cases aus...")
    main_folder_path = filedialog.askdirectory(title="Wähle den Ordner 'Cases'")
    if not main_folder_path:
        print("Abbruch: Kein Ordner ausgewählt.")
        return
        
    main_folder = Path(main_folder_path)
    all_tables = []
    print(f"Lese CSVs aus '{main_folder.name}' ein (Dies kann kurz dauern)...")

    # 1. ALLE DATEN ROH EINLESEN
    regex_pattern = re.compile(r"area_Element([^_]+)_([^_]+)_time_(.*?)\.csv")
    for file_pfad in main_folder.rglob("*.csv"):
        df_temp = pd.read_csv(file_pfad)
        case_folder = next(p for p in file_pfad.parents if p.name.startswith("Case")).name
        
        hit = regex_pattern.search(file_pfad.name)
        if hit:
            linse = hit.group(1)
            surface_num = hit.group(2)
            time_val = float(hit.group(3))
            
            mapping = {'0':'bottom','1':'bottom','2':'top','3':'top','4':'lateral','5':'support'} if linse in ['1', '3'] else {'0':'bottom','1':'bottom','2':'top','3':'lateral','4':'support'}
                
            df_temp['case_id'] = case_folder
            df_temp['Linsen_art'] = linse
            df_temp['Surface'] = mapping.get(surface_num, 'unknown')
            df_temp['Time'] = time_val
            
            # HTC berechnen
            df_temp = df_temp[(df_temp['T'] != 0) & (df_temp['contact area'] != 0)]
            df_temp['HTC'] = df_temp['Q'] / (df_temp['contact area'] * df_temp['T'])
            df_temp = df_temp.drop(columns=['Q'])
            
            all_tables.append(df_temp)

    if not all_tables:
        print("Keine gültigen CSV-Dateien gefunden.")
        return

    df_final = pd.concat(all_tables, ignore_index=True)
    del all_tables
    gc.collect()

    # Speicheroptimierung: Downcast float64 zu float32
    for col in df_final.select_dtypes(include=['float64']).columns:
        df_final[col] = df_final[col].astype('float32')

    
    # 2. PHASEN-ERKENNUNG
    
    print("Starte Phasen-Erkennung (Heatup vs. Cooldown)...")
    
    # Für die Phase schauen wir uns die echte TEMPERATUR (T) an
    df_trend_T = df_final.groupby(['case_id', 'Time'])['T'].mean().unstack(level=0)
    
    phasen_zuordnung = {}

    for case in df_trend_T.columns:
        # Temperatur ganz am Anfang vs. ganz am Ende
        t_start = df_trend_T[case].dropna().iloc[0]
        t_ende = df_trend_T[case].dropna().iloc[-1]
        
        # Wenn es am Ende wärmer ist -> Heatup!
        phasen_zuordnung[case] = 'Heatup' if t_ende > t_start else 'Cooldown'


    print("\nAutomatische Phasen-Zuweisung erfolgreich:")
    for c, p in sorted(phasen_zuordnung.items()):
        print(f"   -> {c}: Erkannt als {p}")


    # 3. ZUWEISUNG ANWENDEN UND SPEICHERN
    
    # 1 für Heatup, 0 für Cooldown
    df_final['Phase'] = df_final['case_id'].map(lambda x: 1 if phasen_zuordnung[x] == 'Heatup' else 0)

    # Parquet direkt im aktuellen Ordner (wo das Notebook liegt) ablegen
    speicher_pfad = Path("./linsen_daten_roh.parquet")
    df_final.to_parquet(speicher_pfad, index=False)
    
    # .absolute() zeigt dir in der Konsole den genauen, vollen Pfad an
    print(f"\nFertig! Daten gespeichert unter:\n{speicher_pfad.absolute()}")
    del df_final, df_trend_T
    gc.collect()
    
# ---------------------------------------------------------
# 2. Funktion: Outlier filtern
# ---------------------------------------------------------
def filter_outliers():
    
    if Path("linsen_daten_roh.parquet").exists() and not Path("linsen_daten_clean.parquet").exists():
        print("Lade rohe Daten für Outlier-Filterung...")
        df = pd.read_parquet("linsen_daten_roh.parquet")
        original_zeilen = len(df)
        
        untere_grenze = df.groupby(['case_id', 'Linsen_art', 'Surface'])['HTC'].transform(lambda x: x.quantile(0.01))
        obere_grenze = df.groupby(['case_id', 'Linsen_art', 'Surface'])['HTC'].transform(lambda x: x.quantile(0.99))
        
        maske_sauber = (df['HTC'] >= untere_grenze) & (df['HTC'] <= obere_grenze)
        df_clean = df[maske_sauber].copy()
        
        entfernt = original_zeilen - len(df_clean)
        print(f"Filterung abgeschlossen! Es wurden {entfernt} kaputte Knotenpunkte ({(entfernt/original_zeilen)*100:.2f}%) entfernt.")
        
        ziel_pfad = Path("./linsen_daten_clean.parquet")
        df_clean.to_parquet(ziel_pfad, index=False)
        print(f"Saubere Daten als '{ziel_pfad.absolute()}' gespeichert.")
        
        del df, df_clean
        gc.collect()
    elif Path("linsen_daten_clean.parquet").exists():
        print("Übersprungen: 'linsen_daten_clean.parquet' existiert bereits.")
    else:
        print("Fehler: 'linsen_daten_roh.parquet' nicht gefunden. Bitte zuerst Schritt 1 ausführen.")


# ---------------------------------------------------------
# Skript-Ausführung
# ---------------------------------------------------------
if __name__ == "__main__":
    erstelle_parquet()
    filter_outliers() 
    print("Parquet-Dateien erfolgreich erstellt!")