# outils communs aux 2 modèles
    # - charger les features (X) et les targets (y)
    # - couper train / test dans le temps
    # - préparer les colonnes (encodage du texte)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Features numériques (déjà des nombres)
FEATURES_NUM = [
    "delai_promis", "mois_achat", "jour_semaine", "heure_achat", "delai_validation_h",
    "nb_produits", "nb_vendeurs", "prix_total", "frais_port",
    "poids_total_g", "volume_total_cm3", "nb_photos",
    "meme_etat", "nb_mensualites",
]

FEATURES_TEXTE = ["categorie", "etat_client", "etat_vendeur", "type_paiement"]
FEATURES = FEATURES_NUM + FEATURES_TEXTE

# Date de coupure, avant = entraînement, après = test
DATE_COUPURE = "2018-04-01"


def charger_donnees(chemin="data-clean/olist_ml.csv"):
    return pd.read_csv(chemin, parse_dates=["order_purchase_timestamp"])


def decouper_train_test(df, target):
# coupe dans le temps on apprend sur le passé, on teste sur le futur
    passe = df["order_purchase_timestamp"] < DATE_COUPURE
    X_train, y_train = df.loc[passe, FEATURES], df.loc[passe, target]
    X_test,  y_test  = df.loc[~passe, FEATURES], df.loc[~passe, target]
    return X_train, X_test, y_train, y_test


def creer_preprocesseur():
    # ecode les colonnes texte en colonnes 0/1
    return ColumnTransformer([
        ("texte", OneHotEncoder(handle_unknown="ignore"), FEATURES_TEXTE),
    ], remainder="passthrough")   # passthrough = les colonnes numériques passent sans changement




if __name__ == "__main__":
    df = charger_donnees()
    X_train, X_test, y_train, y_test = decouper_train_test(df, "en_retard")
    print("Train :", X_train.shape, f"- retards {y_train.mean():.1%}")
    print("Test  :", X_test.shape,  f"- retards {y_test.mean():.1%}")