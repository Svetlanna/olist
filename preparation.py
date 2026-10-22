
import pandas as pd

DOSSIER = "data-raw/"


dates = ["order_purchase_timestamp", "order_approved_at",
         "order_delivered_customer_date", "order_estimated_delivery_date"]
orders    = pd.read_csv(DOSSIER + "olist_orders_dataset.csv", parse_dates=dates)  # dates converties dès la lecture
items     = pd.read_csv(DOSSIER + "olist_order_items_dataset.csv")
customers = pd.read_csv(DOSSIER + "olist_customers_dataset.csv")
products  = pd.read_csv(DOSSIER + "olist_products_dataset.csv")
sellers   = pd.read_csv(DOSSIER + "olist_sellers_dataset.csv")
payments  = pd.read_csv(DOSSIER + "olist_order_payments_dataset.csv")



orders = orders[(orders["order_status"] == "delivered")
                & orders["order_delivered_customer_date"].notna()]

# -----------------les target ----------------
# target 1 (classification)  1 = en retard, 0 = à l'heure
orders["en_retard"] = (orders["order_delivered_customer_date"]
                       > orders["order_estimated_delivery_date"]).astype(int)

# target 2 (régression) délai réel en jours (avec décimales)
# .dt.total_seconds() / 86400 transforme une durée en nombre de jours
orders["delai_jours"] = ((orders["order_delivered_customer_date"]
                          - orders["order_purchase_timestamp"])
                         .dt.total_seconds() / 86400)

# ---------------- FEATURES DE TEMPS (connues au moment de l'achat) --------------
achat = orders["order_purchase_timestamp"]
orders["delai_promis"] = (orders["order_estimated_delivery_date"] - achat).dt.total_seconds() / 86400
orders["mois_achat"]   = achat.dt.month
orders["jour_semaine"] = achat.dt.dayofweek          # 0 = lundi, 6 = dimanche
orders["heure_achat"]  = achat.dt.hour
orders["delai_validation_h"] = (orders["order_approved_at"] - achat).dt.total_seconds() / 3600

# -----------deatures du coly (agrégation 1 ligne par commande) ---------------
# on ajoute aux produits achetés les infos produit et vendeur
items = (items
         .merge(products, on="product_id", how="left")
         .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left"))
items["volume_cm3"] = (items["product_length_cm"] * items["product_height_cm"]
                       * items["product_width_cm"])

# groupby + agg = regrouper toutes les lignes d'une même commande
colis = items.groupby("order_id").agg(
    nb_produits  = ("order_item_id", "count"),            # combien de produits
    nb_vendeurs  = ("seller_id", "nunique"),           # combien de vendeurs différents
    prix_total   = ("price", "sum"),
    frais_port   = ("freight_value", "sum"),
    poids_total_g = ("product_weight_g", "sum"),
    volume_total_cm3 = ("volume_cm3", "sum"),
    nb_photos    = ("product_photos_qty", "mean"),
    categorie    = ("product_category_name", "first"),    # catégorie du 1er produit
    etat_vendeur = ("seller_state", "first"),             # État du 1er vendeur
).reset_index()

# ============ 6. FEATURES DE PAIEMENT ============
paiement = payments.groupby("order_id").agg(
    type_paiement  = ("payment_type", "first"),
    nb_mensualites = ("payment_installments", "max"),
).reset_index()

# ============ 7. TOUT RELIER ============
df = (orders
      .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
      .merge(colis,    on="order_id", how="inner")
      .merge(paiement, on="order_id", how="left"))

df = df.rename(columns={"customer_state": "etat_client"})
df["meme_etat"] = (df["etat_client"] == df["etat_vendeur"]).astype(int)

# ============ 8. NETTOYAGE ============
# Valeurs manquantes : texte -> "inconnu", nombres -> médiane
for col in ["categorie", "etat_vendeur", "type_paiement"]:
    df[col] = df[col].fillna("inconnu")
for col in ["delai_validation_h", "poids_total_g", "volume_total_cm3", "nb_photos", "nb_mensualites"]:
    df[col] = df[col].fillna(df[col].median())

# ============ 9. SÉPARER FEATURES (X) ET TARGETS (y) ============
FEATURES = [
    # temps
    "delai_promis", "mois_achat", "jour_semaine", "heure_achat", "delai_validation_h",
    # colis
    "nb_produits", "nb_vendeurs", "prix_total", "frais_port",
    "poids_total_g", "volume_total_cm3", "nb_photos", "categorie",
    # géographie
    "etat_client", "etat_vendeur", "meme_etat",
    # paiement
    "type_paiement", "nb_mensualites",
]
TARGETS = ["en_retard", "delai_jours"]

X        = df[FEATURES]
y_retard = df["en_retard"]     # pour le modèle 1 (classification)
y_delai  = df["delai_jours"]   # pour le modèle 2 (régression)



# On garde aussi la date d'achat : utile pour couper train/test dans le temps
df[["order_id", "order_purchase_timestamp"] + FEATURES + TARGETS].to_csv(
    "data-clean/olist_ml.csv", index=False)

# ============ 10. VÉRIFICATIONS ============
print("Taille de X :", X.shape)
print("Valeurs manquantes dans X :", X.isna().sum().sum())
print(f"\nCommandes en retard : {y_retard.mean():.1%}")
print("\nDélai réel (jours) :")
print(y_delai.describe().round(1))
print("\nDélai promis (jours) :")
print(df["delai_promis"].describe().round(1))
print("\n5 premières lignes de X :")
print(X.head())
print("\nFichier enregistré : data-clean/olist_ml.csv")