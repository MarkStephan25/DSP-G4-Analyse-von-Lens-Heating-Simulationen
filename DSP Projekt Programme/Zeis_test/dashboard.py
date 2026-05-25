import ipywidgets as widgets
from IPython.display import display

def zeige_steuerung(f_etl, f_filter, f_train, f_eval, f_repair):
    header = widgets.HTML(
        value="<h3 style='color: #2c3e50; margin-bottom: 0;'>Digitaler Zwilling - Control Panel</h3>"
    )

    out = widgets.Output(layout=widgets.Layout(
        border='1px solid #bdc3c7', padding='15px', border_radius='8px', width='780px'
    ))

    btn_layout = widgets.Layout(width='145px', height='45px', margin='4px')

    btn_etl = widgets.Button(description=" 1. ETL", layout=btn_layout)
    btn_filter = widgets.Button(description=" 2. Filter", layout=btn_layout)
    btn_train = widgets.Button(description=" 3. Training", layout=btn_layout)
    btn_eval = widgets.Button(description=" 4. Evaluieren", layout=btn_layout)
    btn_repair = widgets.Button(description=" 4.5 Repair", layout=btn_layout)

    # Farben zuweisen
    btn_etl.style.button_color, btn_etl.style.text_color = '#3498db', 'white'
    btn_filter.style.button_color, btn_filter.style.text_color = '#1abc9c', 'white'
    btn_train.style.button_color, btn_train.style.text_color = '#f39c12', 'white'
    btn_eval.style.button_color, btn_eval.style.text_color = '#2ecc71', 'white'
    btn_repair.style.button_color, btn_repair.style.text_color = '#e74c3c', 'white'

    @out.capture(clear_output=True)
    def run_etl(b):
        print("Starte Schritt 1...")
        f_etl()

    @out.capture(clear_output=True)
    def run_filter(b):
        print("Starte Schritt 2...")
        f_filter()

    @out.capture(clear_output=True)
    def run_train(b):
        print("Starte Schritt 3...")
        f_train()

    @out.capture(clear_output=True)
    def run_eval(b):
        print("Starte Schritt 4...")
        f_eval()

    @out.capture(clear_output=True)
    def run_repair(b):
        print("Starte Schritt 4.5...")
        f_repair()

    btn_etl.on_click(run_etl)
    btn_filter.on_click(run_filter)
    btn_train.on_click(run_train)
    btn_eval.on_click(run_eval)
    btn_repair.on_click(run_repair)

    tasten_reihe = widgets.HBox([btn_etl, btn_filter, btn_train, btn_eval, btn_repair])
    display(widgets.VBox([header, tasten_reihe]))
    display(out)