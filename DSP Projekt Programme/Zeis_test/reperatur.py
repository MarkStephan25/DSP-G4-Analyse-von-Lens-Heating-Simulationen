import pandas as pd
import numpy as np
import sympy
import os
import glob      
import gc
from scipy.optimize import curve_fit
from pysr import PySRRegressor
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================
def base_time_curve(t, a, b, c, d, e):
    # Die physikalische Aufheiz-Basis (Doppel-Exponential)
    return a * np.exp(-b * t) + c * np.exp(-d * t) + e

# ==============================================================================
# HAUPTFUNKTION: SPEZIAL-TRAINING
# ==============================================================================
def repariere_modell(ziel_linse, ziel_surface, ziel_case, phase=1, filter_faktor=5.0, erlaube_z_fuer_support=True):
    print(f"\nSTARTE REPARATUR-TRAINING FÜR:")
    print(f"Linse: {ziel_linse} | Surface: {ziel_surface} | Case: {ziel_case}")
    print(f"-> Median-Filter Faktor: {filter_faktor}")
    print(f"-> Z-Achse für Support erlaubt: {erlaube_z_fuer_support}")
    print("-" * 50)

    # 1. Daten laden und sofort radikal filtern (Memory Saving)
    print("Lade CFD-Daten...")
    cols = ['Phase', 'Linsen_art', 'Surface', 'case_id', 'Time', 'x', 'y', 'z', 'HTC']
    df = pd.read_parquet("linsen_daten_clean.parquet", columns=cols)
    df['HTC_abs'] = np.abs(df['HTC'])
    df['Linsen_art'] = df['Linsen_art'].astype(str)

    # Filtern auf die EINE gewünschte Zone
    maske = (df['Phase'] == phase) & (df['Linsen_art'] == str(ziel_linse)) & (df['Surface'] == ziel_surface) & (df['case_id'] == ziel_case)
    df_train = df[maske].copy()
    
    del df
    gc.collect()

    if df_train.empty:
        print("Keine Daten für diese Kombination gefunden. Abbruch!")
        return

    # 2. Ausreißer filtern (Mit dynamischem Faktor!)
    print("🧹 Wende Median-Filter an...")
    zeit_kurve = df_train.groupby('Time')['HTC_abs'].transform('median')
    obergrenze = np.maximum(zeit_kurve * filter_faktor, 10.0)
    df_train = df_train[df_train['HTC_abs'] <= obergrenze].copy()
    print(f"-> {len(df_train)} saubere Datenpunkte für das Training übrig.")

    # 3. Geometrie-Features berechnen (Das 3D Upgrade)
    print("Berechne 3D-Features...")
    df_train['t_skaliert'] = df_train['Time'] / 20000.0

    # FIX: Sollen wir dem Support die Z-Achse erlauben?
    if ziel_surface == 'support' and not erlaube_z_fuer_support:
        df_train['r_skaliert'] = 0.0
        df_train['z_skaliert'] = 0.0
    else:
        df_train['r_skaliert'] = np.sqrt(df_train['x']**2 + df_train['y']**2) * 1000.0
        z_max = df_train['z'].max()
        df_train['z_skaliert'] = np.abs(z_max - df_train['z']) * 1000.0

    df_train['r_norm'] = df_train['r_skaliert'] / 100.0
    df_train['z_norm'] = df_train['z_skaliert'] / 100.0
    df_train['r_norm_sq'] = df_train['r_norm']**2
    df_train['z_norm_sq'] = df_train['z_norm']**2
    df_train['r_t_interaction'] = df_train['r_norm'] * df_train['t_skaliert']
    df_train['z_t_interaction'] = df_train['z_norm'] * df_train['t_skaliert']

    # 4. Base Curve (Zeit) fitten
    print("Fitte zeitliche Basis-Kurve...")
    t_vals = df_train['t_skaliert'].values
    htc_vals = df_train['HTC_abs'].values

    # Start-Schätzungen (Guess) für a, b, c, d, e
    p0 = [np.mean(htc_vals), 1.0, np.mean(htc_vals)*0.1, 0.1, 0.0]
    
    try:
        base_params, _ = curve_fit(base_time_curve, t_vals, htc_vals, p0=p0, maxfev=5000)
    except Exception as e:
        print(f"Fehler beim Curve-Fit: {e}. Verwende Fallback-Werte.")
        base_params = [np.mean(htc_vals), 0, 0, 0, 0]

    # Berechne, was noch fehlt (Der Multiplikator für PySR)
    htc_base_pred = base_time_curve(t_vals, *base_params)
    y_pysr = htc_vals / (htc_base_pred + 1e-5) # Verhindert Division durch Null
    
    # Ausreißer im Multiplikator kappen (Stabilität für PySR)
    y_pysr = np.clip(y_pysr, -10.0, 10.0)

    # 5. PySR Symbolic Regression
    print("\nStarte PySR KI-Training (Das kann ein paar Minuten dauern)...")
    
    # Nur die 6 räumlichen Features an PySR übergeben (Zeit ist ja schon in der Base-Curve)
    X_pysr = df_train[['r_norm', 'z_norm', 'r_norm_sq', 'z_norm_sq', 'r_t_interaction', 'z_t_interaction']]

    model = PySRRegressor(
        niterations=30,           # Gut für schnelle Reparaturen
        populations=10,
        population_size=33,
        maxsize=15,               # Kompakt halten!
        maxdepth=3,               # Verhindert Monster-Verschachtelungen
        parsimony=0.1,    # Zwingt die KI zu einfachen Formeln
        binary_operators=["+", "-", "*"],
        unary_operators=["exp"],  # Nur e-Funktionen erlaubt
        random_state=42,
        verbosity=0               # Konsole nicht zuspammen
    )

    model.fit(X_pysr, y_pysr)
    
    beste_formel = str(model.sympy())
    print(f"\nPySR hat die Formel gefunden!")
    print(f"🧬 Multiplikator: {beste_formel}")

    # 6. Formel abspeichern (Der intelligente Registry-Manager)
    print("\nUpdate die Registry und lösche alte Versionen...")
    params_str = "|".join(map(str, base_params))
    
    neue_zeile = pd.DataFrame([{
        'Phase': phase,
        'Linse': str(ziel_linse),
        'Surface': ziel_surface,
        'Case': ziel_case,
        'Base_Params': params_str,
        'PySR_Multiplikator': beste_formel
    }])
    
    # Finde alle existierenden CSV-Dateien im Ordner
    alte_dateien = glob.glob("Formel_Registry_*.csv")
    
    if alte_dateien:
        # Alles in einen großen Topf werfen
        dfs = [pd.read_csv(d, sep=";") for d in alte_dateien]
        df_master = pd.concat(dfs, ignore_index=True)
        
        maske_alt = (df_master['Phase'] == phase) & \
                    (df_master['Linse'].astype(str) == str(ziel_linse)) & \
                    (df_master['Surface'] == ziel_surface) & \
                    (df_master['Case'] == ziel_case)
        
        anzahl_geloescht = maske_alt.sum()
        df_master = df_master[~maske_alt] # Behalte alles AUSSER der alten Formel!
        print(f"-> {anzahl_geloescht} alte Version(en) dieser Formel gefunden und vernichtet.")
        
        # Neue Formel anhängen
        df_master = pd.concat([df_master, neue_zeile], ignore_index=True)
        
        # Aufräumen: Lösche das Dateichaos im Ordner
        for d in alte_dateien:
            try:
                os.remove(d)
            except Exception as e:
                print(f"Konnte {d} nicht löschen (vielleicht noch geöffnet?): {e}")
                
    else:
        # Falls der Ordner komplett leer ist
        df_master = neue_zeile
        
    # Alles sauber in EINE einzige Master-Datei speichern
    master_name = "Formel_Registry_Master.csv"
    df_master.to_csv(master_name, sep=";", index=False)
    
    print(f"Registry erfolgreich geupdatet! Dein Ordner ist aufgeräumt.")
    print(f"Alle Formeln liegen jetzt sicher in: {master_name}")


if __name__ == "__main__":
    
    # ==============================================================================
    # DEIN STEUERPULT FÜR REPARATUREN 
    # ==============================================================================
    
    # Reparatur 1: Linse 2 Support (Problem: Jackson-Pollock Rauschen)
    # Wir setzen den Filter auf 7.0 und schalten die Z-Achse an!
    repariere_modell(
        ziel_linse='2',
        ziel_surface='support',
        ziel_case='Case1',
        filter_faktor=7.0,             # Erlaubt extremere Hitze-Spitzen
        erlaube_z_fuer_support=True    # Erlaubt der KI zu lernen, dass es oben heißer ist!
    )
    
    # Reparatur 2: Linse 3 Lateral (Problem: Hotspot an der Kante)
    # Filter leicht lockern, damit die 434 HTC voll getroffen werden.
    repariere_modell(
        ziel_linse='3',
        ziel_surface='lateral',
        ziel_case='Case1',
        filter_faktor=6.0,
        erlaube_z_fuer_support=False   # Für Lateral irrelevant
    )