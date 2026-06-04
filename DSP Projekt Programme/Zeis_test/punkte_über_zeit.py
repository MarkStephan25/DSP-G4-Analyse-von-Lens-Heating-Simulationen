import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from validierung import DigitalerZwillingBatch 

warnings.filterwarnings('ignore')

def plot_htc_zeitverlauf_mit_modell(ziel_linse='3', ziel_surface='top', ziel_case='Case_2', ziel_phase=0):
    print(f"Lade Daten für Zeitverlauf: Linse {ziel_linse} | {ziel_surface} | {ziel_case} | Phase {ziel_phase}...")
    df = pd.read_parquet("linsen_daten_clean.parquet")
    
    # --------------------------------------------------------
    # 1. Daten dynamisch filtern
    # --------------------------------------------------------
    maske = (
        (df['Linsen_art'].astype(str) == str(ziel_linse)) & 
        (df['Surface'] == str(ziel_surface)) & 
        (df['case_id'] == str(ziel_case)) &
        (df['Phase'] == ziel_phase)
    )
    df_sub = df[maske].copy()
    
    if df_sub.empty:
        print("Keine Daten für diese Kombination gefunden!")
        return
        
    df_sub['HTC_abs'] = np.abs(df_sub['HTC'])
    
    # --------------------------------------------------------
    # 2. Modell-Vorhersage berechnen
    # --------------------------------------------------------
    print("Berechne KI-Vorhersagen für diesen Case...")
    zwilling = DigitalerZwillingBatch()
    df_sub['HTC_Pred'] = zwilling.berechne_vorhersage_schnell(
        df_sub, phase=ziel_phase, linse=str(ziel_linse), surface=ziel_surface, case_id=ziel_case
    )
    
    # Filtere Punkte raus, bei denen PySR evtl. abgestürzt ist
    df_sub = df_sub.dropna(subset=['HTC_Pred'])
    
    # --------------------------------------------------------
    # 3. Kurven berechnen (Mean & Max pro Zeitsprung)
    # --------------------------------------------------------
    df_zeit = df_sub.groupby('Time').agg(
        mean_cfd=('HTC_abs', 'mean'),
        max_cfd=('HTC_abs', 'max'),
        mean_pred=('HTC_Pred', 'mean'),
        max_pred=('HTC_Pred', 'max')
    ).reset_index()
    
    # --------------------------------------------------------
    # 4. Das Plotting (UPGRADE: Subplots & Downsampling)
    # --------------------------------------------------------
    # Wir nehmen nur 10.000 zufällige Punkte für den Hintergrund (verhindert den "Farbwand"-Effekt)
    df_scatter = df_sub.sample(n=min(len(df_sub), 10000), random_state=42)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
    phasen_name = "Heatup (1)" if ziel_phase == 1 else "Cooldown (0)"
    fig.suptitle(f"KI vs. Realität im Zeitverlauf | Linse {ziel_linse} | {ziel_surface.capitalize()} | {ziel_case} | {phasen_name}", 
                 fontsize=16, fontweight='bold', y=0.98)

    # --- A. OBERER PLOT: Die echten CFD Daten ---
    ax1 = axes[0]
    ax1.scatter(df_scatter['Time'], df_scatter['HTC_abs'], 
                color='#7f7f7f', alpha=0.15, s=12, edgecolor='none', label='Rohdaten CFD (10k Sample)')
    ax1.plot(df_zeit['Time'], df_zeit['mean_cfd'], 
             color='#d62728', linewidth=3.5, label='Echte CFD (Mean)')
    ax1.plot(df_zeit['Time'], df_zeit['max_cfd'], 
             color='#ff7f0e', linewidth=2, linestyle='--', label='Echte CFD (Max/Peak)')
    
    ax1.set_title("1. Die Realität (Echte CFD-Daten)", fontweight='bold')
    ax1.set_ylabel("Wärmeübergang (HTC)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    leg1 = ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=11, ncol=3)
    for lh in leg1.legend_handles: lh.set_alpha(1)

    # --- B. UNTERER PLOT: Die Modell Vorhersagen ---
    ax2 = axes[1]
    ax2.scatter(df_scatter['Time'], df_scatter['HTC_Pred'], 
                color='#1f77b4', alpha=0.15, s=12, edgecolor='none', label='Modell Vorhersage (10k Sample)')
    ax2.plot(df_zeit['Time'], df_zeit['mean_pred'], 
             color='#2ca02c', linewidth=3.5, label='KI Modell (Mean)')
    ax2.plot(df_zeit['Time'], df_zeit['max_pred'], 
             color='#17becf', linewidth=2, linestyle='--', label='KI Modell (Max/Peak)')
    
    ax2.set_title("2. Der Digitale Zwilling (Modell-Vorhersage)", fontweight='bold')
    ax2.set_xlabel("Zeit [s]", fontsize=12)
    ax2.set_ylabel("Wärmeübergang (HTC)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    leg2 = ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=11, ncol=3)
    for lh in leg2.legend_handles: lh.set_alpha(1)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_htc_zeitverlauf_mit_modell(ziel_linse='3', ziel_surface='support', ziel_case='Case1', ziel_phase=1)