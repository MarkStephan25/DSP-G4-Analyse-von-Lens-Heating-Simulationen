import numpy as np
# Wir importieren die Klasse aus deiner diagnose.py Datei
from diagnose import DigitalerZwilling 

print("Lade Digitalen Zwilling in den Arbeitsspeicher...")
# Das dauert einen Bruchteil einer Sekunde. Danach ist er einsatzbereit!
zwilling = DigitalerZwilling()

def berechne_htc_fuer_koordinate(x_meter, y_meter, zeit_sekunden, case_id='Case_1', surface='top'):
    """
    Diese Funktion übersetzt deine 3D-Koordinaten für das KI-Modell.
    """
    # 1. Wir übersetzen X und Y in den Radius (in Millimetern)
    r_mm = np.sqrt(x_meter**2 + y_meter**2) * 1000.0
    
    # 2. Wir bestimmen die Zone (Kern oder Rand)
    # (Trennlinie anpassen, falls die Linsen-Grenze woanders liegt als bei 50mm)
    zone = "Zone_2" if r_mm <= 50.0 else "Zone_3"
    
    # 3. Wir fragen das KI-Modell!
    htc = zwilling.berechne_punkt(
        t_raw=zeit_sekunden, 
        r_raw=r_mm, 
        phase=1, 
        linse='1', 
        surface=surface, 
        zone=zone, 
        case_id=case_id
    )
    
    return htc

# ==========================================
# 🚀 BEISPIEL-ABFRAGEN IN DEINEM NEUEN CODE
# ==========================================

# Beispiel 1: Ein Punkt mitten im Laserstrahl (X=0, Y=0) nach 150 Sekunden
htc_mitte = berechne_htc_fuer_koordinate(x_meter=0.0, y_meter=0.0, zeit_sekunden=150.0)
print(f"HTC im Zentrum: {htc_mitte:.2f}")

# Beispiel 2: Ein Punkt weit am Rand (z.B. X=0.08m, Y=0.0m -> Radius = 80mm) nach 10 Sekunden (Heißester Punkt)
htc_rand = berechne_htc_fuer_koordinate(x_meter=0.08, y_meter=0.0, zeit_sekunden=10.0)
print(f"HTC am Linsenrand: {htc_rand:.2f}")

# Beispiel 3: Abfrage für einen völlig anderen Case ("Case_2")
htc_anderer_case = berechne_htc_fuer_koordinate(x_meter=0.023933845333826045, y_meter=0.11117653393452283, zeit_sekunden=0.0018407605570608912, case_id='Case2')
print(f"HTC am Linsenrand in Case_2: {htc_anderer_case:.2f}")