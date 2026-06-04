import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#  DER GENIALE SCHACHZUG: Wir importieren den fehlerfreien Zwilling!
from validierung import DigitalerZwillingBatch 

warnings.filterwarnings('ignore')

# ==============================================================================
# MAIN: 4-PANEL DASHBOARD (Zonenfrei & DRY)
# ==============================================================================
def plotte_diagnose(test_linse='1', test_surface='top', test_phase=1, test_case='Case_1'):
    print(f"-> Lade Rohdaten für Diagnose (Nur {test_case})...")
    df = pd.read_parquet("linsen_daten_clean.parquet")
    
    maske = (df['Surface'] == test_surface) & (df['Linsen_art'].astype(str) == str(test_linse)) & (df['Phase'] == test_phase) & (df['case_id'] == test_case)
    df_sub = df[maske].copy()
    
    if df_sub.empty:
        verfuegbare_cases = df[(df['Surface'] == test_surface) & (df['Linsen_art'].astype(str) == str(test_linse))]['case_id'].unique()
        print(f" Keine Daten für {test_case} gefunden!")
        print(f" Gefundene Cases für diese Linse: {list(verfuegbare_cases)}")
        return
        
    df_sub['HTC_abs'] = np.abs(df_sub['HTC'])
    
    zeit_kurve = df_sub.groupby('Time')['HTC_abs'].transform('median')
    obergrenze = np.maximum(zeit_kurve * 5.0, 10.0)
    df_sauber = df_sub[df_sub['HTC_abs'] <= obergrenze].copy()
    
    # Geometrie-Variablen berechnen (wird für Plot 4 als X-Achse gebraucht)
    if test_surface == 'support':
        df_sauber['r_skaliert'] = 0.0
        df_sauber['z_skaliert'] = 0.0
    else:
        df_sauber['r_skaliert'] = np.sqrt(df_sauber['x']**2 + df_sauber['y']**2) * 1000.0
        z_max = df_sauber['z'].max()
        df_sauber['z_skaliert'] = np.abs(z_max - df_sauber['z']) * 1000.0
    
    #  WIR NUTZEN DEN IMPORTIERTEN ZWILLING
    zwilling = DigitalerZwillingBatch()

    print("-> Berechne Modell-Vorhersagen für gesamten Datensatz...")
    # Die neue, fehlerfreie und schnelle Vektorisierungs-Methode ohne Zonen
    df_sauber['HTC_Pred'] = zwilling.berechne_vorhersage_schnell(df_sauber, test_phase, str(test_linse), test_surface, test_case)
    
    df_sauber = df_sauber.dropna(subset=['HTC_Pred'])
    
    if df_sauber.empty:
        print(" Konnte keine Vorhersage berechnen. Gibt es ein Modell in der CSV für diesen Case?")
        return
        
    df_sauber['Residuum'] = df_sauber['HTC_abs'] - df_sauber['HTC_Pred']

    # ========================================================
    #  VISUALISIERUNG: 2x2 Raster (4 Plots)
    # ========================================================
    print("-> Erstelle Dashboard...")
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    phasen_name = "Heatup (1)" if test_phase == 1 else "Cooldown (0)"
    fig.suptitle(f"Digitaler Zwilling Dashboard | {phasen_name} | Linse {test_linse} | {test_surface.capitalize()} | {test_case}", 
                 fontsize=18, fontweight='bold', y=0.98)

    # PLOT 1 (Oben Links): Ist vs Soll
    ax1 = axes[0, 0]
    sns.scatterplot(x=df_sauber['HTC_abs'], y=df_sauber['HTC_Pred'], ax=ax1, alpha=0.15, color="#1f77b4", edgecolor="none")
    min_val, max_val = df_sauber['HTC_abs'].min(), df_sauber['HTC_abs'].max()
    ax1.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2.5, label='Perfekte Vorhersage')
    ax1.set_title("Ist-Soll-Vergleich (Gesamte Simulation)", fontweight='bold', pad=12)
    ax1.set_xlabel("CFD Simulation (Echter HTC)")
    ax1.set_ylabel("Modell Vorhersage")
    ax1.legend()

    # PLOT 2 (Oben Rechts): Residuen
    ax2 = axes[0, 1]
    sns.scatterplot(x=df_sauber['HTC_Pred'], y=df_sauber['Residuum'], ax=ax2, alpha=0.15, color="#2ca02c", edgecolor="none")
    ax2.axhline(y=0, color='#d62728', linestyle='--', linewidth=2.5)
    ax2.set_title("Residuendiagramm (Fehler-Streuung)", fontweight='bold', pad=12)
    ax2.set_xlabel("Modell Vorhersage")
    ax2.set_ylabel("Abweichung (Ist minus Soll)")

    # PLOT 3 (Unten Links): Fehlerverteilung
    ax3 = axes[1, 0]
    sns.histplot(df_sauber['Residuum'], kde=True, ax=ax3, bins=50, color="#9467bd", line_kws={'linewidth': 2.5})
    ax3.set_title("Fehlerverteilung (Wie oft irrt sich das Modell?)", fontweight='bold', pad=12)
    ax3.set_xlabel("Abweichung (Residuum)")
    ax3.set_ylabel("Häufigkeit")

    # --------------------------------------------------------
    # PLOT 4 (Unten Rechts): Raum-Profil (Sanity Check)
    # --------------------------------------------------------
    ax4 = axes[1, 1]
    max_time = df_sauber.groupby('Time')['HTC_abs'].mean().idxmax()
    df_tmax = df_sauber[df_sauber['Time'] == max_time].copy()
    
    if test_surface == 'lateral':
        x_col = 'z_skaliert'
        x_label = "Abstand von oben (Z-Achse) [mm]"
    else:
        x_col = 'r_skaliert'
        x_label = "Radius r [mm]"

    df_tmax = df_tmax.sort_values(by=x_col)
    
    ax4.scatter(df_tmax[x_col], df_tmax['HTC_abs'], color='#7f7f7f', alpha=0.4, label=f'Echte CFD (Heißester Pkt T={max_time:.0f})', s=30, edgecolor="none")
    ax4.plot(df_tmax[x_col], df_tmax['HTC_Pred'], color='#ff7f0e', linewidth=4, label='Modell Vorhersage')

    ax4.set_title("Raum-Profil (Physikalischer Sanity Check)", fontweight='bold', pad=12)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel("Wärmeübergang (HTC)")
    ax4.legend(frameon=True, shadow=True, loc='upper right')

    sns.despine()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Teste es direkt mit einem Case!
    plotte_diagnose(test_linse='1', test_surface='support', test_phase=1, test_case='Case1')