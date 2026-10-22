import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from ML import charger_donnees, decouper_train_test, creer_preprocesseur

# 1. Où MLflow enregistre tout : un petit fichier base de données
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Olist - retard de livraison")   # le "classeur" des expériences


# 2. Les données
df = charger_donnees()
X_train, X_test, y_train, y_test = decouper_train_test(df, target="en_retard")


# 3. Une expérience (un "run") par valeur de n_estimators
for n_estimators in [50, 100, 200]:

    with mlflow.start_run(run_name=f"rf_{n_estimators}_arbres"):

        # --- Paramètres du modèle ---
        params = {
            "n_estimators": n_estimators,   # nombre d'arbres dans la forêt
            "max_depth": 12,                # profondeur max de chaque arbre
            "class_weight": "balanced",     # donne plus de poids aux retards (rares : 8 %)
            "random_state": 42,             # pour avoir le même résultat à chaque lancement
        }

        # Pipeline = préparation des colonnes + modèle, collés ensemble
        modele = Pipeline([
            ("preparation", creer_preprocesseur()),
            ("foret", RandomForestClassifier(**params, n_jobs=-1)),
        ])

        # --- Entraînement ---
        modele.fit(X_train, y_train)

        # --- Prédiction sur le test ---
        y_pred = modele.predict(X_test)

        # --- Métriques ---
        accuracy  = accuracy_score(y_test, y_pred)    # % de bonnes réponses
        recall    = recall_score(y_test, y_pred)      # % des vrais retards trouvés
        precision = precision_score(y_test, y_pred)   # % d'alertes "retard" justes
        f1        = f1_score(y_test, y_pred)          # équilibre recall / precision

        # --- Enregistrement dans MLflow ---
        mlflow.log_params(params)                     # tous les paramètres d'un coup
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)

        # Le modèle lui-même + son inscription dans le Model Registry
        mlflow.sklearn.log_model(
            sk_model=modele,
            name="modele",
            input_example=X_train.head(3),            # exemple d'entrée (MLflow en déduit le format)
            registered_model_name="retard_livraison", # nom dans le registre
            serialization_format="cloudpickle",       # format de sauvegarde adapté aux forêts d arbres
        )

        print(f"n_estimators={n_estimators:>3} | accuracy={accuracy:.3f} | "
              f"recall={recall:.3f} | precision={precision:.3f} | f1={f1:.3f}")