import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotte_symmetrie_check(ziel_linse='1', ziel_case='Case_1'):
    print(f"Lade Daten für Symmetrie-Check (Linse: {ziel_linse} | Case: {ziel_case})...")
    df = pd.read_parquet("linsen_daten_clean.parquet")

    #  DYNAMISCHER FILTER: Nur noch Linse und Case! (Phase ist an Case gebunden)
    maske = (
        (df['Linsen_art'].astype(str) == str(ziel_linse)) & 
        (df['case_id'] == str(ziel_case))
    )
    df_gefiltert = df[maske].copy()

    if df_gefiltert.empty:
        print(f"❌ FEHLER: Keine Daten für Linse {ziel_linse} und Case '{ziel_case}' gefunden!")
        return

    df_gefiltert['HTC_abs'] = np.abs(df_gefiltert['HTC'])


    erkannte_phase = df_gefiltert['Phase'].iloc[0]
    phasen_name = "Heatup (1)" if erkannte_phase == 1 else "Cooldown (0)"

    # Wir suchen den HEISSESTEN Zeitpunkt!
    max_time = df_gefiltert.groupby('Time')['HTC_abs'].mean().idxmax()
    df_zeit = df_gefiltert[df_gefiltert['Time'] == max_time].copy()
    print(f"-> Erkannte Phase: {phasen_name}")
    print(f"-> Zeichne heißesten Zeitpunkt bei T = {max_time:.0f}s")

    # Für die laterale Fläche berechnen wir den Winkel (0 bis 360 Grad), um den Zylinder abzurollen
    df_zeit['Winkel'] = np.arctan2(df_zeit['y'], df_zeit['x']) * (180 / np.pi)

    # ========================================================
    #  4 PLOTS EINRICHTEN (2x2 Raster)
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Symmetrie-Check | Linse {ziel_linse} | {ziel_case} | {phasen_name} (T={max_time:.0f}s)", 
                 fontsize=18, fontweight='bold', y=0.98)

    # Einheitliche Farbskala für ALLE Plots
    vmin = df_zeit['HTC_abs'].min()
    vmax = df_zeit['HTC_abs'].max()

    # --------------------------------------------------------
    # 1. Oben Links: TOP
    # --------------------------------------------------------
    df_top = df_zeit[df_zeit['Surface'] == 'top']
    if not df_top.empty:
        sc1 = axes[0, 0].scatter(df_top['x'], df_top['y'], c=df_top['HTC_abs'], cmap='turbo', s=15, vmin=vmin, vmax=vmax)
        axes[0, 0].axis('equal')
    axes[0, 0].set_title("1. Top (Ansicht von Oben)", fontweight='bold')
    axes[0, 0].set_xlabel("X-Koordinate")
    axes[0, 0].set_ylabel("Y-Koordinate")

    # --------------------------------------------------------
    # 2. Oben Rechts: BOTTOM
    # --------------------------------------------------------
    df_bottom = df_zeit[df_zeit['Surface'] == 'bottom']
    if not df_bottom.empty:
        axes[0, 1].scatter(df_bottom['x'], df_bottom['y'], c=df_bottom['HTC_abs'], cmap='turbo', s=15, vmin=vmin, vmax=vmax)
        axes[0, 1].axis('equal')
    axes[0, 1].set_title("2. Bottom (Ansicht von Unten)", fontweight='bold')
    axes[0, 1].set_xlabel("X-Koordinate")
    axes[0, 1].set_ylabel("Y-Koordinate")

    # --------------------------------------------------------
    # 3. Unten Links: LATERAL (Abgerollter Zylinder)
    # --------------------------------------------------------
    df_lat = df_zeit[df_zeit['Surface'] == 'lateral']
    if not df_lat.empty:
        axes[1, 0].scatter(df_lat['Winkel'], df_lat['z'], c=df_lat['HTC_abs'], cmap='turbo', s=15, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("3. Lateral (Zylinder abgerollt: Winkel vs. Z)", fontweight='bold')
    axes[1, 0].set_xlabel("Winkel auf dem Zylinder [Grad]")
    axes[1, 0].set_ylabel("Z-Koordinate")

    # --------------------------------------------------------
    # 4. Unten Rechts: SUPPORT
    # --------------------------------------------------------
    df_sup = df_zeit[df_zeit['Surface'] == 'support']
    if not df_sup.empty:
        axes[1, 1].scatter(df_sup['x'], df_sup['y'], c=df_sup['HTC_abs'], cmap='turbo', s=40, vmin=vmin, vmax=vmax, marker='s')
        axes[1, 1].axis('equal')
    axes[1, 1].set_title("4. Support (Halterungspins)", fontweight='bold')
    axes[1, 1].set_xlabel("X-Koordinate")
    axes[1, 1].set_ylabel("Y-Koordinate")

    # --------------------------------------------------------
    # GEMEINSAME COLORBAR
    # --------------------------------------------------------
    try:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sc1, cax=cbar_ax, label='Wärmeübergangskoeffizient (HTC)')
    except:
        pass 

    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.show()

# =====================================================================
#  HIER STEUERST DU DEN PLOT!
# =====================================================================
if __name__ == "__main__":
    plotte_symmetrie_check(ziel_linse='3', ziel_case='Case1')