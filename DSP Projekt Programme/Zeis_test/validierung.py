import pandas as pd
import numpy as np
import sympy
import glob
import warnings
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# KLASSE: DIGITALER ZWILLING (High-Speed Batch Evaluator - Pure NumPy)
# ==============================================================================
class DigitalerZwillingBatch:
    def __init__(self, ziel_linse=None, ziel_surface=None, ziel_case=None):
        print("📖 Lade Formel-Registry-Dateien...")
        
        dateien = glob.glob("Formel_Registry_*.csv")
        if not dateien: 
            raise FileNotFoundError("❌ Keine CSV-Dateien gefunden!")
        
        dfs = [pd.read_csv(d, sep=";") for d in dateien]
        self.registry = pd.concat(dfs, ignore_index=True)
        
        # Registry nach Wunsch filtern
        if ziel_linse:
            self.registry = self.registry[self.registry['Linse'].astype(str) == str(ziel_linse)]
        if ziel_surface:
            self.registry = self.registry[self.registry['Surface'] == ziel_surface]
        if ziel_case:
            self.registry = self.registry[self.registry['Case'] == ziel_case]
            
        if self.registry.empty:
            raise ValueError("❌ Für diese Filterkombination gibt es keine Formeln in den CSV-Dateien!")

        self.formel_cache = {}
        
        # 🔥 ALLE 7 FEATURES DEFINIERT (Verhindert den SymPy-Crash bei z_t_interaction)
        self.t_sym, self.r_n_sym, self.z_n_sym, self.r_sq_sym, self.z_sq_sym, self.r_t_sym, self.z_t_sym = sympy.symbols(
            't_skaliert r_norm z_norm r_norm_sq z_norm_sq r_t_interaction z_t_interaction'
        )
        print(f"✅ {len(self.registry)} relevante Formeln für diesen Lauf gefunden!\n")

    def hole_formel(self, phase, linse, surface, case_id):
        key = (phase, str(linse), surface, case_id)
        if key in self.formel_cache: 
            return self.formel_cache[key]
            
        mask = (self.registry['Phase'] == phase) & (self.registry['Linse'].astype(str) == str(linse)) & (self.registry['Surface'] == surface) & (self.registry['Case'] == case_id)
        if mask.sum() == 0: 
            return None
        
        row = self.registry[mask].iloc[0]
        base_params = tuple(map(float, row['Base_Params'].split('|')))
        expr = sympy.parsing.sympy_parser.parse_expr(str(row['PySR_Multiplikator']))
        
        # Lambdify mit allen 7 Variablen für rasend schnelle Ausführung
        mult_func = sympy.lambdify((self.t_sym, self.r_n_sym, self.z_n_sym, self.r_sq_sym, self.z_sq_sym, self.r_t_sym, self.z_t_sym), expr, modules=['numpy'])
        
        self.formel_cache[key] = (base_params, mult_func)
        return self.formel_cache[key]

    def berechne_vorhersage_schnell(self, df_sub, phase, linse, surface, case_id):
        formel_data = self.hole_formel(phase, linse, surface, case_id)
        if formel_data is None: 
            return np.full(len(df_sub), np.nan)

        (a, b, c, d, e), mult_func = formel_data
        
        # 🔥 PERFORMANCE-UPGRADE: Reine NumPy Arrays statt Pandas Spalten!
        t = df_sub['Time'].values / 20000.0
        
        if surface == 'support':
            r_n = np.zeros_like(t)
            z_n = np.zeros_like(t)
        else:
            x = df_sub['x'].values
            y = df_sub['y'].values
            z = df_sub['z'].values
            
            r_skaliert = np.sqrt(x**2 + y**2) * 1000.0
            z_max = np.max(z)
            z_skaliert = np.abs(z_max - z) * 1000.0
            
            r_n = r_skaliert / 100.0
            z_n = z_skaliert / 100.0

        # Alle Features als reine C-Level NumPy-Arrays
        r_sq = r_n**2
        z_sq = z_n**2
        i = r_n * t
        z_t = z_n * t  # Das 7. Feature

        htc_base = a * np.exp(-b * t) + c * np.exp(-d * t) + e
        
        try:
            # Funktion mit allen 7 Parametern aufrufen
            multiplikator = mult_func(t, r_n, z_n, r_sq, z_sq, i, z_t)
            return htc_base * multiplikator
        except Exception as err:
            print(f"\n❌ Fehler bei Berechnung: {err}")
            return np.full(len(t), np.nan)

# ==============================================================================
# MAIN: AUTOMATISCHE VALIDIERUNG ALLER MODELLE
# ==============================================================================
def validiere_alles(ziel_linse=None, ziel_surface=None, ziel_case=None):
    print("🚀 Starte globale Modell-Validierung (High-Speed Memory Optimized)...\n")
    
    zwilling = DigitalerZwillingBatch(ziel_linse, ziel_surface, ziel_case)
    
    print("-> Lade CFD-Rohdaten (Das kann einen Moment dauern)...")
    columns_needed = ['Phase', 'Linsen_art', 'Surface', 'case_id', 'Time', 'x', 'y', 'z', 'HTC']
    df_cfd = pd.read_parquet("linsen_daten_clean.parquet", columns=columns_needed)
    
    df_cfd['HTC_abs'] = np.abs(df_cfd['HTC'])
    df_cfd['Linsen_art'] = df_cfd['Linsen_art'].astype(str)
    
    # 🔥 SPEICHER-RETTUNG: Wirf sofort weg, was wir nicht prüfen wollen!
    print("-> Filtere unnötige CFD-Daten sofort aus dem Arbeitsspeicher...")
    if ziel_linse:
        df_cfd = df_cfd[df_cfd['Linsen_art'] == str(ziel_linse)]
    if ziel_surface:
        df_cfd = df_cfd[df_cfd['Surface'] == ziel_surface]
    if ziel_case:
        df_cfd = df_cfd[df_cfd['case_id'] == ziel_case]
        
    gc.collect() # PC aufräumen lassen
    
    # 🔥 TURBO-SUCHE: Erstelle das Inhaltsverzeichnis (Hash-Map)
    print("-> Erstelle High-Speed Suchindex...")
    cfd_index = df_cfd.groupby(['Phase', 'Linsen_art', 'Surface', 'case_id'])
    
    kombinationen = zwilling.registry[['Phase', 'Linse', 'Surface', 'Case']].drop_duplicates().reset_index(drop=True)
    ergebnisse = []
    total_modelle = len(kombinationen)
    
    print(f"\n⚙️ Starte Auswertung von {total_modelle} Modellen...\n")

    for idx, row in kombinationen.iterrows():
        phase = row['Phase']
        linse = str(row['Linse'])
        surface = row['Surface']
        case = row['Case']
        
        print(f"[{idx+1}/{total_modelle}] Prüfe: Phase {phase} | Linse {linse} | {surface:<7} | {case} ...", end=" ", flush=True)
        
        # Millisekunden-Abfrage über Hash-Map
        try:
            df_sub = cfd_index.get_group((phase, linse, surface, case)).copy()
        except KeyError:
            print("❌ Keine CFD Daten für diese Kombination.")
            continue
            
        # Median-Filterung
        zeit_kurve = df_sub.groupby('Time')['HTC_abs'].transform('median')
        obergrenze = np.maximum(zeit_kurve * 5.0, 10.0)
        df_sub = df_sub[df_sub['HTC_abs'] <= obergrenze]
            
        # Vorhersage
        df_sub['HTC_Pred'] = zwilling.berechne_vorhersage_schnell(df_sub, phase, linse, surface, case)
        df_sub = df_sub.dropna(subset=['HTC_Pred', 'HTC_abs'])
        
        if df_sub.empty:
            print("❌ Fehler bei Vorhersage (Leeres Dataframe).")
            continue
            
        # Metriken berechnen
        y_true = df_sub['HTC_abs'].values
        y_pred = df_sub['HTC_Pred'].values
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        max_err = np.max(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        print(f"✅ R²: {r2:+.3f} | RMSE: {rmse:<5.1f} | MAE: {mae:<5.2f}")
        
        ergebnisse.append({
            'Phase': phase,
            'Linse': linse,
            'Surface': surface,
            'Case': case,
            'Datenpunkte': len(y_true),
            'Max_HTC_CFD': np.max(y_true),
            'R_Quadrat': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Max_Error': max_err
        })
        
        # Nach jedem Modell den RAM gnadenlos leeren!
        del df_sub
        gc.collect()

    # ==============================================================================
    # REPORT SPEICHERN UND AUSWERTEN
    # ==============================================================================
    if ergebnisse:
        df_report = pd.DataFrame(ergebnisse)
        df_report = df_report.sort_values(by='R_Quadrat', ascending=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        dateiname = f"Validierungs_Report_{timestamp}.csv"
        df_report.to_csv(dateiname, index=False, sep=";")
        
        print("\n" + "="*60)
        print("🏆 REPORT ZUSAMMENFASSUNG")
        print("="*60)
        print(f"Durchschnittliches R² über alle Modelle: {df_report['R_Quadrat'].mean():.4f}")
        print(f"Durchschnittlicher MAE (Fehler in HTC):  {df_report['MAE'].mean():.2f}")
        
        print("\n🌟 TOP 3 MODELLE (Die Besten):")
        print(df_report[['Case', 'Surface', 'Phase', 'R_Quadrat', 'MAE']].head(3).to_string(index=False))
        
        print("\n⚠️ FLOP 3 MODELLE (Hier gibt es noch Probleme):")
        print(df_report[['Case', 'Surface', 'Phase', 'R_Quadrat', 'MAE']].tail(3).to_string(index=False))
        
        print(f"\n📂 Kompletter Report gespeichert als: {dateiname}")
    else:
        print("\n❌ Es konnten keine Modelle validiert werden.")

if __name__ == "__main__":
    
    # 🎛️ DEIN STEUERPULT 🎛️
    # Ändere diese Werte, um blitzschnell nur bestimmte Dinge zu testen.
    # Setze einen Wert auf None, wenn du ihn nicht filtern möchtest.
    
    validiere_alles(
        ziel_linse=None,      # z.B. '1', '2' oder None
        ziel_case='Case1',   # z.B. 'Case1', 'Case2' oder None
        ziel_surface=None    # z.B. 'top', 'lateral' oder None
    )