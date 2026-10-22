import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


def choisir(nom_modele, metrique, plus_grand_est_mieux):
    meilleure_version, meilleur_score = None, None

    for version in client.search_model_versions(f"name='{nom_modele}'"):
        run = client.get_run(version.run_id)
        score = run.data.metrics[metrique]
        print(f"  {nom_modele} v{version.version} : {metrique} = {score:.3f}")

        est_meilleur = (meilleur_score is None
                        or (plus_grand_est_mieux and score > meilleur_score)
                        or (not plus_grand_est_mieux and score < meilleur_score))
        if est_meilleur:
            meilleure_version, meilleur_score = version.version, score

    client.set_registered_model_alias(nom_modele, "champion", meilleure_version)
    print(f"  -> champion = version {meilleure_version} ({metrique} = {meilleur_score:.3f})\n")


print("Modèle 1 : retard (on veut le F1 le plus GRAND)")
choisir("retard_livraison", "f1", plus_grand_est_mieux=True)

print("Modèle 2 : délai (on veut la MAE la plus PETITE)")
choisir("delai_livraison", "mae_jours", plus_grand_est_mieux=False)