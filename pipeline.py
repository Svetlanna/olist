import subprocess   # permet de lancer un autre programme (ici, nos scripts Python)
import sys          # sys.executable = le Python de ton venv
import time
from pathlib import Path


ETAPES = [
    ("1. Préparer les données",       "preparation.py"),
    ("2. Entraîner le modèle retard", "entrainer_retard.py"),
    ("3. Entraîner le modèle délai",  "entrainer_delai.py"),
    ("4. Choisir les champions",      "choisir_champion.py"),
    ("5. Prédire une nouvelle commande", "predire.py"),
]


def lancer_etape(nom, script):

    print(f"\n{'=' * 60}\n{nom}  ({script})\n{'=' * 60}")
    debut = time.time()

    resultat = subprocess.run([sys.executable, script])

    duree = time.time() - debut
    if resultat.returncode != 0:          # 0 = succès, autre chose = erreur
        print(f"\nÉCHEC de l'étape « {nom} » après {duree:.0f} s.")
        print("Le pipeline s'arrête : corrige l'erreur ci-dessus, puis relance.")
        sys.exit(1)
    print(f"-> OK en {duree:.0f} s")


if __name__ == "__main__":

    if not Path("data-raw/olist_orders_dataset.csv").exists():
        sys.exit("Erreur : les CSV Olist doivent être dans le dossier data-raw/")

    Path("data-clean").mkdir(exist_ok=True)   # crée le dossier de sortie s'il manque

    debut_total = time.time()
    for nom, script in ETAPES:
        lancer_etape(nom, script)

    print(f"\nPipeline terminé en {time.time() - debut_total:.0f} s.")
    print("Ouvre MLflow : mlflow ui --backend-store-uri sqlite:///mlflow.db")