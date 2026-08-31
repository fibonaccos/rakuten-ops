# Rakuten — plateforme d'inférence

Classification des produits Rakuten en 27 catégories. Le front Streamlit appelle
une passerelle FastAPI authentifiée par JWT, qui interroge un service d'inférence
chargeant son modèle depuis MLflow et journalise chaque prédiction en base.
Prometheus et Grafana supervisent l'ensemble ; Locust sert de banc de charge.

Deux réseaux Docker : `frontend` exposé, `backend` interne. Seuls l'API, MLflow,
Grafana et le front publient un port.

## Démarrer

```bash
cp .env.example .env      # puis renseigner chaque valeur
docker compose up -d
```

Front sur http://localhost:8501, API sur http://localhost:8000/docs, MLflow sur
http://localhost:5001, Grafana sur http://localhost:3000. Le banc de charge est
derrière un profil : `docker compose --profile locust up locust`.

Les images de la plateforme (`api`, `inference`, `database`, `streamlit`,
`locust`) sont construites depuis le dépôt, pas tirées d'un registre : ce que la
stack déploie est donc toujours le code de la branche courante. Le premier
démarrage prend quelques minutes, les suivants réutilisent le cache.

Deux échecs courants au premier lancement :

- `Bind for 0.0.0.0:8501 failed: port is already allocated` — un autre projet
  occupe 8000, 8501, 5001 ou 3000. `docker ps` pour voir qui, puis arrêter la
  stack concernée.
- `dependency failed to start: container inference-service is unhealthy` — le
  service d'inférence télécharge le modèle depuis MLflow avant de répondre. Si
  cela dépasse `start_period`, l'API renonce à démarrer. Relancer
  `docker compose up -d` suffit, le modèle étant alors en cache.

## Développer

Dépendances gérées par [uv](https://docs.astral.sh/uv/), en groupes qui
correspondent aux images : `api`, `inference`, `streamlit`, `ml`, `benchmark`,
plus `dev` pour l'outillage.

```bash
make install      # uv sync --all-groups
make lint         # ruff
make test         # toutes les suites
```

Pour lancer le front seul :
`uv run --group streamlit streamlit run services/streamlit/app.py`.

## Tests

Chaque service est une image autonome avec ses propres requirements, et leurs
modules portent les mêmes noms (`main`, `routes`, `services`) : chaque suite
tourne donc dans son propre processus.

```bash
make test-stack       # docker-compose.yaml conforme à .env.example
make test-api         # authentification, autorisations, contrat d'inférence
make test-inference   # routes d'inférence, sur un modèle bouchon
make test-ui          # client HTTP et rendu de chaque page Streamlit
make test-ml          # pipeline d'inférence et parité avec l'entraînement
```

`tests/ml/test_cleaning_parity.py` mérite une mention : il vérifie que le
nettoyage appliqué au service (`model/pipeline.py`) produit exactement le même
texte que celui appliqué à l'entraînement (`src/rakuten/data/clean_data.py`).
Une divergence entre les deux est un décalage entraînement/service : le modèle
reçoit un texte qu'il n'a jamais vu, la qualité chute, et rien ne lève d'erreur.

## Intégration continue

`.github/workflows/ci.yml` tourne sur `main`, `master` et les branches
`dev-**`, `feature/**`, `fix/**` : lint, puis une suite par groupe de
dépendances, puis la construction des cinq images.

Les règles ruff sont épinglées dans `pyproject.toml` plutôt qu'étendues, la
sélection par défaut de l'outil s'élargissant d'une version à l'autre. Le tri
des imports est volontairement exclu : l'activer réécrirait tout le dépôt.
Quand l'équipe le voudra, `uv run ruff check --select I --fix .`.

## Environnement

Les variables sont documentées dans `.env.example`, préfixées par service
(`RAKUTEN__API__*`, `RAKUTEN__INFERENCE__*`, `RAKUTEN__LOCUST__*`,
`RAKUTEN__GRAFANA__*`). Chaque service les valide au démarrage : une variable
manquante arrête le conteneur au lieu de le laisser servir à moitié configuré.
`.env` n'est jamais versionné, et un test le vérifie.
