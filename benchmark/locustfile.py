import json
import logging
import os
import random
import time

import pandas as pd
import psycopg2
import yaml
from locust import HttpUser, events, task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rakuten-locust")


def load_config():
    path = os.environ.get("SIMULATION_CONFIG", "/benchmark/config/simulation.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_credential_pool(retries=10, delay=3):
    db_url = os.environ["DATABASE_URL"]
    pwd_path = os.environ["PASSWORD_FILE"]

    with open(pwd_path, encoding="utf-8") as f:
        plaintext = json.load(f)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(db_url, connect_timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT username, role FROM users")
            rows = cur.fetchall()
            cur.close()
            conn.close()

            pool = [
                {"username": u, "password": plaintext[u], "role": r}
                for u, r in rows
                if u in plaintext
            ]
            if not pool:
                raise RuntimeError(
                    "aucun utilisateur de la table 'users' n'a de mot de passe "
                    "correspondant dans PASSWORD_FILE"
                )
            log.info(
                "Pool d'identifiants charge: %d comptes (%d admin, %d user)",
                len(pool),
                sum(1 for c in pool if c["role"] == "admin"),
                sum(1 for c in pool if c["role"] == "user"),
            )
            return pool
        except Exception as exc:  # noqa: BLE001 - on veut logger puis retry
            last_err = exc
            log.warning("Connexion DB echouee (tentative %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)

    raise RuntimeError(f"Impossible de charger le pool d'identifiants depuis la base: {last_err}")


def load_samples_pool():
    path = os.environ["DATA_FILE"]
    df = pd.read_parquet(path)
    if "designation" not in df.columns:
        raise RuntimeError(f"colonne 'designation' absente de {path} (colonnes: {list(df.columns)})")

    has_description = "description" in df.columns
    records = []
    for row in df.itertuples(index=False):
        designation = getattr(row, "designation")
        description = getattr(row, "description") if has_description else None
        if pd.isna(description):
            description = None
        records.append({"designation": designation, "description": description})

    if not records:
        raise RuntimeError(f"{path} ne contient aucune ligne exploitable")

    log.info("Pool d'echantillons d'inference charge: %d lignes", len(records))
    return records


CONFIG = load_config()
CREDENTIAL_POOL = load_credential_pool()
SAMPLES_POOL = load_samples_pool()


def pick_credential(role=None):
    pool = [c for c in CREDENTIAL_POOL if role is None or c["role"] == role]
    if not pool:
        raise RuntimeError(f"aucun compte disponible pour role={role!r}")
    return random.choice(pool)


def sample_one():
    return dict(SAMPLES_POOL[random.randrange(len(SAMPLES_POOL))])


def sample_batch(n=None):
    n = n or random.randint(50, 500)
    n = min(n, len(SAMPLES_POOL))
    return [dict(SAMPLES_POOL[i]) for i in random.sample(range(len(SAMPLES_POOL)), k=n)]


def make_wait_time(mean, sigma):
    def _wait_time(self):
        return max(0.1, random.gauss(mean, sigma))
    return _wait_time


class RakutenUser(HttpUser):
    abstract = True

    def on_start(self):
        self.token = None
        self.username = None
        self.password = None

    def _auth_header(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _login(self, name="/auth/login"):
        with self.client.post(
            "/auth/login",
            data={"username": self.username, "password": self.password},
            name=name,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                self.token = resp.json()["access_token"]
                resp.success()
                return True
            resp.failure(f"login KO: {resp.status_code}")
            self.token = None
            return False

    def _protected_call(self, method, url, expect_ok=(200,), name=None, **kwargs):
        name = name or url
        headers = kwargs.pop("headers", {})
        headers.update(self._auth_header())

        with self.client.request(
            method, url, headers=headers, name=name, catch_response=True, **kwargs
        ) as resp:
            if resp.status_code in expect_ok:
                resp.success()
                return
            if resp.status_code == 401:
                resp.success()  # 401 transitoire attendu a ce stade du cycle de vie du token
                if self._login():
                    headers.update(self._auth_header())
                    with self.client.request(
                        method, url, headers=headers, name=name, catch_response=True, **kwargs
                    ) as retry:
                        if retry.status_code in expect_ok:
                            retry.success()
                        else:
                            retry.failure(f"echec apres relogin: {retry.status_code}")
                return
            resp.failure(f"status inattendu: {resp.status_code}")


class LegitBaseUser(RakutenUser):
    abstract = True

    def on_start(self):
        super().on_start()
        cred = pick_credential(role="user")
        self.username, self.password = cred["username"], cred["password"]
        self._login()

    @task(30)
    def predict_single(self):
        self._protected_call("POST", "/predict/single", json=sample_one(), name="/predict/single")

    @task(3)
    def check_profile(self):
        self._protected_call("GET", "/auth/me", name="/auth/me")

    @task(2)
    def browse_models(self):
        self._protected_call("GET", "/models", name="/models")

    @task(5)
    def browse_current_model(self):
        self._protected_call("GET", "/models/current", name="/models/current")


class LegitLightUser(LegitBaseUser):
    pass


class LegitModerateUser(LegitBaseUser):
    pass


class LegitHeavyUser(LegitBaseUser):
    pass


class LegitAdminUser(LegitBaseUser):
    def on_start(self):
        RakutenUser.on_start(self)

        cred = pick_credential(role="admin")
        self.username = cred["username"]
        self.password = cred["password"]
        self._login()

    @task(5)
    def predict_single(self):
        self._protected_call("POST", "/predict/single", json=sample_one(), name="/predict/single")

    @task(30)
    def predict_batch(self):
        self._protected_call("POST", "/predict/batch", json=sample_batch(), name="/predict/batch")

    @task(15)
    def admin_models(self):
        self._protected_call(
            "GET",
            "/models",
            name="/models [admin]"
        )

    @task(10)
    def admin_current_model(self):
        self._protected_call(
            "GET",
            "/models/current",
            name="/models/current [admin]"
        )

    @task(3)
    def admin_profile(self):
        self._protected_call(
            "GET",
            "/auth/me",
            name="/auth/me [admin]"
        )



class UnauthenticatedUser(RakutenUser):
    @task(6)
    def try_predict_no_auth(self):
        with self.client.post(
            "/predict/single", json=sample_one(), name="/predict/single [no-auth]", catch_response=True
        ) as resp:
            resp.success() if resp.status_code == 401 else resp.failure(
                f"attendu 401, obtenu {resp.status_code}"
            )

    @task(4)
    def try_me_no_auth(self):
        with self.client.get("/auth/me", name="/auth/me [no-auth]", catch_response=True) as resp:
            resp.success() if resp.status_code == 401 else resp.failure(
                f"attendu 401, obtenu {resp.status_code}"
            )

    @task(2)
    def try_models_no_auth(self):
        with self.client.get("/models", name="/models [no-auth]", catch_response=True) as resp:
            resp.success() if resp.status_code == 401 else resp.failure(
                f"attendu 401, obtenu {resp.status_code}"
            )

    @task(1)
    def check_health(self):
        with self.client.get("/health", name="/health", catch_response=True) as resp:
            resp.success() if resp.status_code == 200 else resp.failure(
                f"attendu 200, obtenu {resp.status_code}"
            )


class UnregisteredUser(RakutenUser):
    @task
    def attempt_login_unknown_user(self):
        ghost = random.choice(CONFIG["unregistered_usernames"])
        with self.client.post(
            "/auth/login",
            data={"username": ghost["username"], "password": ghost["password"]},
            name="/auth/login [unregistered]",
            catch_response=True,
        ) as resp:
            resp.success() if resp.status_code == 401 else resp.failure(
                f"attendu 401, obtenu {resp.status_code}"
            )


class ExpiredTokenUser(RakutenUser):
    def on_start(self):
        super().on_start()
        cred = pick_credential(role="user")
        self.username, self.password = cred["username"], cred["password"]
        self._expiry_logged = False
        self._login()

    def _call(self, method, url, name, **kwargs):
        with self.client.request(
            method, url, headers=self._auth_header(), name=name, catch_response=True, **kwargs
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 401:
                resp.success()  # comportement attendu: token expire, jamais rafraichi
                if not self._expiry_logged:
                    log.info("[ExpiredTokenUser] expiration du JWT detectee pour %s", self.username)
                    self._expiry_logged = True
            else:
                resp.failure(f"status inattendu: {resp.status_code}")

    @task(6)
    def predict_single(self):
        self._call("POST", "/predict/single", "/predict/single [may-be-expired]", json=sample_one())

    @task(2)
    def check_profile(self):
        self._call("GET", "/auth/me", "/auth/me [may-be-expired]")


CLASS_MAP = {
    "legit_light": LegitLightUser,
    "legit_moderate": LegitModerateUser,
    "legit_heavy": LegitHeavyUser,
    "unauthenticated": UnauthenticatedUser,
    "unregistered": UnregisteredUser,
    "expired_token": ExpiredTokenUser,
    "legit_admin": LegitAdminUser,
}

def _apply_config_to_user_classes():
    for key, cls in CLASS_MAP.items():
        cls.weight = CONFIG["population"][key]
        wt = CONFIG["wait_time"][key]
        cls.wait_time = make_wait_time(wt["mean"], wt["sigma"])
 
 
_apply_config_to_user_classes()


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    log.info("=== Simulation Rakuten API ===")
    log.info(
        "users=%s spawn_rate=%s run_time=%s",
        CONFIG["run"]["users"], CONFIG["run"]["spawn_rate"], CONFIG["run"]["run_time"],
    )
    log.info(
        "pool identifiants=%d comptes | pool echantillons=%d lignes",
        len(CREDENTIAL_POOL), len(SAMPLES_POOL),
    )
    for key, cls in CLASS_MAP.items():
        wt = CONFIG["wait_time"][key]
        log.info(
            "  - %-20s weight=%-3s wait~lognorm(mean=%ss, sigma=%s)",
            key, cls.weight, wt["mean"], wt["sigma"],
        )
