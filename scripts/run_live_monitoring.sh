#!/usr/bin/env bash
# Wrapper pour lancer retrain_trigger_live.py depuis cron (ou tout
# ordonnanceur). Cron ne connaît pas les variables d'environnement de ton
# shell interactif -- ce script les charge explicitement depuis un fichier
# local, jamais commité.
#
# Mise en place, une seule fois :
#   1. Copie ce fichier tel quel (ne rien modifier ici).
#   2. Crée scripts/.env.monitoring (voir le modèle plus bas), rempli avec
#      tes vraies valeurs. Ce fichier ne doit JAMAIS être commité --
#      vérifie qu'il est bien dans .gitignore.
#   3. Rends ce script exécutable : chmod +x scripts/run_live_monitoring.sh
#   4. Programme-le dans cron (voir instructions données à côté).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="scripts/.env.monitoring"
if [ ! -f "$ENV_FILE" ]; then
    echo "Fichier $ENV_FILE introuvable. Crée-le d'abord (voir le commentaire en haut de ce script)." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/live_monitoring_$(date +%Y%m%d).log"

{
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    uv run python src/monitoring/retrain_trigger_live.py
    echo
} >> "$LOG_FILE" 2>&1
