import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")


modele_retard = mlflow.sklearn.load_model("models:/retard_livraison@champion")
modele_delai  = mlflow.sklearn.load_model("models:/delai_livraison@champion")

# 2. une nouvelle commande, avec EXACTEMENT les mêmes colonnes que l'entraînement
nouvelle_commande = pd.DataFrame([{
    "delai_promis": 25.0,          # le site promet 25 jours
    "mois_achat": 11,              # novembre (Black Friday)
    "jour_semaine": 4,             # vendredi
    "heure_achat": 21,
    "delai_validation_h": 0.5,     # paiement validé en 30 minutes
    "nb_produits": 2,
    "nb_vendeurs": 1,
    "prix_total": 180.0,           # en R$
    "frais_port": 45.0,
    "poids_total_g": 8000.0,       # 8 kg
    "volume_total_cm3": 60000.0,
    "nb_photos": 2.0,
    "meme_etat": 0,                # vendeur et client dans des États différents
    "nb_mensualites": 3.0,
    "categorie": "moveis_decoracao",   # meubles et déco
    "etat_client": "BA",               # Bahia (nord-est, loin)
    "etat_vendeur": "SP",              # São Paulo
    "type_paiement": "credit_card",
}])

# 3. Prédictions
retard      = modele_retard.predict(nouvelle_commande)[0]
proba       = modele_retard.predict_proba(nouvelle_commande)[0][1]   # probabilité de la classe 1
delai_jours = modele_delai.predict(nouvelle_commande)[0]

print("=== Prédiction pour la nouvelle commande ===")
print(f"Retard prévu      : {'OUI' if retard == 1 else 'NON'} (probabilité {proba:.0%})")
print(f"Délai prévu       : {delai_jours:.1f} jours")
print(f"Délai promis      : {nouvelle_commande['delai_promis'][0]:.0f} jours")