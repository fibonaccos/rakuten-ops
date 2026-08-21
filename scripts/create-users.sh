#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set=api_user="$RAKUTEN__API__DATABASE_USER" \
  --set=api_password="$RAKUTEN__API__DATABASE_PASSWORD" \
  <<'SQL'

CREATE USER :"api_user" WITH PASSWORD :'api_password';

GRANT USAGE ON SCHEMA public TO :"api_user";

GRANT SELECT
ON TABLE users
TO :"api_user";

GRANT SELECT, INSERT, UPDATE
ON TABLE inference
TO :"api_user";

GRANT USAGE, SELECT
ON SEQUENCE inference_inference_id_seq
TO :"api_user";


CREATE USER :"locust_user" WITH PASSWORD :'locust_password';

GRANT USAGE ON SCHEMA public TO :"locust_user";

GRANT SELECT
ON TABLE users
TO :"locust_user";

GRANT USAGE, SELECT
ON SEQUENCE inference_inference_id_seq
TO :"locust_user";

SQL