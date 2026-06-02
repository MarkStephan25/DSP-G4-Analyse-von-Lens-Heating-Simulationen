import pandas as pd
import numpy as np
import warnings
from pysr import PySRRegressor
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

def double_exp(t, a, b, c, d, e):
    return a * np.exp(-b * t) + c * np.exp(-d * t) + e

def master_pipeline_dynamic(ziel_linse=None, ziel_surface=None, ziel_case=None):
    
    print(f"🚀 Starte UNIVERSELLE 3D-Pipeline mit LIVE-SPEICHERUNG...")
    print(f"Filter -> Linse: {ziel_linse or 'ALLE'} | Surface: {ziel_surface or 'ALLE'} | Case: {ziel_case or 'ALLE'}")
    
    # --------------------------------------------------------
    # 1. Dateinamen vorher festlegen
    # --------------------------------------------------------
    name_parts = ["Formel_Registry"]
    if ziel_linse: name_parts.append(f"L{ziel_linse}")
    if ziel_surface: name_parts.append(f"{ziel_surface}")
    if ziel_case: name_parts.append(f"{ziel_case}")
    
    dateiname = "Formel_Registry_ALLE_DATEN.csv" if len(name_parts) == 1 else "_".join(name_parts) + ".csv"
    
    # --------------------------------------------------------
    # 2. Leere Datei mit Spaltenköpfen anlegen
    # --------------------------------------------------------
    spalten = ['Phase', 'Linse', 'Surface', 'Case', 'Base_Params', 'Base_Formel', 'PySR_Multiplikator']
    pd.DataFrame(columns=spalten).to_csv(dateiname, index=False, sep=";")
    print(f"📁 Datei '{dateiname}' erstellt. Starte Live-Streaming...\n")
    
    df = pd.read_parquet("linsen_daten_clean.parquet")
    
    # Daten filtern
    maske = pd.Series(True, index=df.index)
    if ziel_linse is not None: maske &= (df['Linsen_art'].astype(str) == str(ziel_linse))
    if ziel_surface is not None: maske &= (df['Surface'] == str(ziel_surface))
    if ziel_case is not None: maske &= (df['case_id'] == str(ziel_case))
        
    df = df[maske].copy()
    if df.empty:
        print(f"❌ Keine Daten gefunden. Beende.")
        return

    df['t_skaliert'] = df['Time'] / 20000.0
    df['HTC_abs'] = np.abs(df['HTC'])
    
    # 🔥 NEU: Globales Maximum für jede Phase/Linse-Kombination berechnen (für die 2%-Schwelle)
    max_werte = df.groupby(['Phase', 'Linsen_art'])['HTC_abs'].max().to_dict()
    
    gruppen = df.groupby(['Phase', 'Linsen_art', 'Surface', 'case_id'])
    
    for (phase, linse, surface, case_id), df_sub in gruppen:
        phase_name = "Heatup (1)" if phase == 1 else "Cooldown (0)"
        print(f"\n==================================================")
        print(f"🔄 Bearbeite: Linse {linse} | {surface} | Case: {case_id} | Phase: {phase_name}")
        
        # --------------------------------------------------------
        # 🔥 DYNAMISCHER COLD-SKIP (2%-Regel)
        # --------------------------------------------------------
        globales_max = max_werte.get((phase, linse), 100.0) 
        dynamische_schwelle = globales_max * 0.02  # Alles unter 2% des Maximums ist "kalt"
        
        if df_sub['HTC_abs'].max() < dynamische_schwelle:
            print(f"🥶 Max HTC ({df_sub['HTC_abs'].max():.1f}) unter der 2%-Schwelle ({dynamische_schwelle:.1f}). Überspringe PySR...")
            
            # Fehlerfreies Erstellen der Zeile als Konstante
            neue_zeile = pd.DataFrame([{
                'Phase': phase, 'Linse': linse, 'Surface': surface, 'Case': case_id, 
                'Base_Params': f"{df_sub['HTC_abs'].mean():.4f}|0|0|0|0",
                'Base_Formel': str(df_sub['HTC_abs'].mean()),
                'PySR_Multiplikator': "1.0"
            }])
            neue_zeile.to_csv(dateiname, mode='a', header=False, index=False, sep=";")
            print("      💾 [Erfolgreich als Konstante live gespeichert]")
            continue

        # --------------------------------------------------------
        # 3D-FEATURE-ENGINEERING (R UND Z)
        # --------------------------------------------------------
        if surface == 'support':
            df_sub['r_skaliert'] = 0.0
            df_sub['z_skaliert'] = 0.0
        else:
            df_sub['r_skaliert'] = np.sqrt(df_sub['x']**2 + df_sub['y']**2) * 1000.0
            z_max = df_sub['z'].max()
            df_sub['z_skaliert'] = np.abs(z_max - df_sub['z']) * 1000.0
            
        zeit_kurve = df_sub.groupby('Time')['HTC_abs'].transform('median')
        obergrenze = np.maximum(zeit_kurve * 5.0, 10.0)
        df_sauber = df_sub[df_sub['HTC_abs'] <= obergrenze].copy()
        
        df_sauber['r_gerundet'] = df_sauber['r_skaliert'].round(1)
        df_sauber['z_gerundet'] = df_sauber['z_skaliert'].round(1)
        df_sauber = df_sauber.groupby(['Time', 't_skaliert', 'r_gerundet', 'z_gerundet'])['HTC_abs'].mean().reset_index()
        df_sauber = df_sauber.rename(columns={'r_gerundet': 'r_skaliert', 'z_gerundet': 'z_skaliert'})

        # --------------------------------------------------------
        # 1. THERMODYNAMIK-BASIS
        # --------------------------------------------------------
        df_0d = df_sauber.groupby('t_skaliert')['HTC_abs'].mean().reset_index().dropna()
        if df_0d.empty or len(df_0d) < 10: continue
            
        t_val, y_val = df_0d['t_skaliert'].values, df_0d['HTC_abs'].values
        diff = np.max(y_val) - np.min(y_val)
        p0 = [diff * 0.7, 15.0, diff * 0.3, 1.5, np.min(y_val)] 
        
        try:
            popt, _ = curve_fit(double_exp, t_val, y_val, p0=p0, maxfev=50000, bounds=(0, np.inf))
            a_f, b_f, c_f, d_f, e_f = popt
            formel_base = f"({a_f:.3f}*exp(-{b_f:.3f}*t_skaliert) + {c_f:.3f}*exp(-{d_f:.3f}*t_skaliert) + {e_f:.3f})"
            df_sauber['HTC_Base'] = double_exp(df_sauber['t_skaliert'].values, *popt)
        except Exception:
            df_sauber['HTC_Base'] = y_val.mean()
            formel_base, a_f, b_f, c_f, d_f, e_f = str(y_val.mean()), y_val.mean(), 0, 0, 0, 0

        # --------------------------------------------------------
        # 2. PySR TRAINING
        # --------------------------------------------------------
        df_sauber['Multiplikator_Target'] = df_sauber['HTC_abs'] / df_sauber['HTC_Base']
        
        if len(df_sauber) < 3000:
            print(f"      -> Kleine Datenmenge ({len(df_sauber)} Punkte): Nehme alle (Kein Sampling).")
            df_train = df_sauber.copy()
        else:
            print(f"      -> Große Datenmenge ({len(df_sauber)} Punkte): Erstelle Sample...")
            df_sauber['Klasse'] = pd.qcut(df_sauber['HTC_abs'], q=10, labels=False, duplicates='drop')
            df_train = pd.concat([df_k.sample(n=min(len(df_k), 300), random_state=42) for _, df_k in df_sauber.groupby('Klasse')])
        
        df_train['Multiplikator_Target'] = np.clip(df_train['HTC_abs'] / df_train['HTC_Base'], 0.0, 30.0) 
        
        df_train['r_norm'] = df_train['r_skaliert'] / 100.0
        df_train['z_norm'] = df_train['z_skaliert'] / 100.0
        df_train['r_norm_sq'] = df_train['r_norm']**2
        df_train['z_norm_sq'] = df_train['z_norm']**2
        df_train['r_t_interaction'] = df_train['r_norm'] * df_train['t_skaliert']
        
        # 🔥 NEU: Die dynamische Weiche für Lateral vs. Top/Bottom
        if surface == 'lateral':
            df_train['z_t_interaction'] = df_train['z_norm'] * df_train['t_skaliert']
            feature_liste = ['r_norm', 'z_norm', 't_skaliert', 'r_norm_sq', 'z_norm_sq', 'r_t_interaction', 'z_t_interaction']
            aktuelle_parsimony = 0.01
        else:
            feature_liste = ['r_norm', 't_skaliert', 'r_norm_sq', 'r_t_interaction']
            aktuelle_parsimony = 0.1

        X = df_train[feature_liste]
        y = df_train['Multiplikator_Target'].values
        
        # PySR mit der dynamischen Parsimony
        modell = PySRRegressor(
            procs=8, 
            niterations=3000, 
            binary_operators=["+", "-", "*"], 
            unary_operators=["exp", "square"], 
            maxsize=15,             
            maxdepth=3,             
            parsimony=aktuelle_parsimony,  # Passt sich automatisch der Fläche an!
            constraints={'exp': 3, 'square': 2},
            random_state=42,        
            model_selection="best", 
            verbosity=0
        )
        
        try:
            modell.fit(X, y, weights=df_train['HTC_abs'].values)
            formel_mult = str(modell.sympy()) 
            print(f"✅ Erfolgreich trainiert!")
            
            neue_zeile = pd.DataFrame([{
                'Phase': phase, 'Linse': linse, 'Surface': surface, 'Case': case_id,
                'Base_Params': f"{a_f:.4f}|{b_f:.4f}|{c_f:.4f}|{d_f:.4f}|{e_f:.4f}",
                'Base_Formel': formel_base, 'PySR_Multiplikator': formel_mult
            }])
            neue_zeile.to_csv(dateiname, mode='a', header=False, index=False, sep=";")
            print("      💾 [Erfolgreich live gespeichert]")
            
        except Exception as e:
            print(f"❌ Fehler bei PySR: {e}")

    print(f"\n🎉 Alles fertig. '{dateiname}' ist komplett!")


if __name__ == "__main__":
    master_pipeline_dynamic(ziel_linse='3',ziel_case='Case1')
