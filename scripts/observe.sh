#!/usr/bin/env bash
# One view of the whole pipeline.
#
# Three systems hold the answer and none of them can see the other two: Kafka
# knows the backlog, Airflow knows what is running, and the catalogue knows
# what actually landed. Diagnosing a slow run means reading all three and
# holding them side by side, which is a thing worth having a command for.
#
#   ./scripts/observe.sh          human-readable
#   ./scripts/observe.sh --json   the same numbers, for anything else
set -euo pipefail
cd "$(dirname "$0")/.."

JSON=${1:-}

lag_rows() {
  docker compose exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh \
    --bootstrap-server kafka:9092 --describe --all-groups 2>/dev/null \
    | awk '/books\./ {print $1"\t"$2"\t"$3"\t"$5"\t"$6}'
}

airflow_sql() {
  docker compose exec -T airflow-db psql -U airflow -d airflow -Atc "$1" 2>/dev/null
}

catalogue() {
  docker compose exec -T airflow-scheduler python /tmp/_observe_catalogue.py "$@" 2>/dev/null
}

docker compose cp scripts/observe_catalogue.py airflow-scheduler:/tmp/_observe_catalogue.py >/dev/null 2>&1

if [ "$JSON" = "--json" ]; then
  printf '{"lag":['
  lag_rows | awk -F'\t' '{printf "%s{\"group\":\"%s\",\"topic\":\"%s\",\"partition\":%s,\"end\":%s,\"lag\":%s}", (NR>1?",":""), $1,$2,$3,$4,$5}'
  printf '],"catalogue":'
  catalogue --json
  printf '}\n'
  exit 0
fi

echo "── consumers ─────────────────────────────────────────────"
printf "  %-24s %-14s %-4s %10s %8s\n" GROUP TOPIC PART END LAG
lag_rows | while IFS=$'\t' read -r g t p e l; do
  printf "  %-24s %-14s %-4s %10s %8s\n" "$g" "$t" "$p" "$e" "$l"
done
TOTAL=$(lag_rows | awk -F'\t' '$1 ~ /load/ {s+=$5} END {print s+0}')
echo "  load backlog: ${TOTAL} messages"

echo
echo "── airflow ───────────────────────────────────────────────"
airflow_sql "SELECT '  '||dag_id||' | '||state||' | '||coalesce(start_date::text,'-') FROM dag_run ORDER BY start_date DESC NULLS LAST LIMIT 5"
echo "  in flight: $(airflow_sql "SELECT count(*) FROM dag_run WHERE state IN ('running','queued')")"

echo
echo "── catalogue ─────────────────────────────────────────────"
catalogue
