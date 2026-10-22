import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from ML import charger_donnees, decouper_train_test, creer_preprocesseur

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Olist - delai de livraison")

df = charger_donnees()
X_train, X_test, y_train, y_test = decouper_train_test(df, target="delai_jours")

for n_estimators in [50, 100, 200]:

    with mlflow.start_run(run_name=f"rf_{n_estimators}_arbres"):

        params = {
            "n_estimators": n_estimators,
            "max_depth": 12,
            "min_samples_leaf": 5,# au moins 5 commandes par feuille évite d'apprendre par cœur
            "random_state": 42,
        }

        modele = Pipeline([
            ("preparation", creer_preprocesseur()),
            ("foret", RandomForestRegressor(**params, n_jobs=-1)),
        ])
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)       # erreur moyenne en jours
        rmse = root_mean_squared_error(y_test, y_pred)   # punit plus les grosses erreurs
        r2   = r2_score(y_test, y_pred)                  # 1 = parfait, 0 = pas mieux que la moyenne

        mlflow.log_params(params)
        mlflow.log_metric("mae_jours", mae)
        mlflow.log_metric("rmse_jours", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            sk_model=modele,
            name="modele",
            input_example=X_train.head(3),
            registered_model_name="delai_livraison",
            serialization_format="cloudpickle",       # format de sauvegarde adapté aux forêts d arbres
        )

        print(f"n_estimators={n_estimators:>3} | MAE={mae:.2f} j | RMSE={rmse:.2f} j | R2={r2:.3f}")