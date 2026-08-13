import json
import pathlib

PG = lambda uid: {"type": "postgres", "uid": uid}
PROM = {"type": "prometheus", "uid": "prometheus"}
LAG = 'sum(kafka_consumergroup_lag{consumergroup="book-pipeline-load"})'
DRAIN = f"-deriv({LAG}[10m:30s])"
_id = iter(range(200, 400))


def sql(ds, q, fmt="table"):
    return [{"refId": "A", "format": fmt, "rawQuery": True, "rawSql": q, "datasource": PG(ds)}]


def promq(expr, legend=""):
    return [{"refId": "A", "expr": expr, "legendFormat": legend, "datasource": PROM}]


def stat(title, targets, ds, x, y, w, h, unit="short", steps=None, mode="none", desc=""):
    return {
        "type": "stat",
        "title": title,
        "description": desc,
        "id": next(_id),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": ds,
        "targets": targets,
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": mode,
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": 0,
                "thresholds": {
                    "mode": "absolute",
                    "steps": steps or [{"color": "text", "value": None}],
                },
            },
            "overrides": [],
        },
    }


def ts(title, targets, ds, x, y, w, h, unit="short", desc="", fill=14):
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "id": next(_id),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": ds,
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 2,
                    "fillOpacity": fill,
                    "gradientMode": "opacity",
                    "showPoints": "never",
                    "spanNulls": True,
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def table(title, targets, ds, x, y, w, h, desc=""):
    return {
        "type": "table",
        "title": title,
        "description": desc,
        "id": next(_id),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": ds,
        "targets": targets,
        "options": {"showHeader": True, "cellHeight": "sm"},
        "fieldConfig": {
            "defaults": {"custom": {"align": "auto", "filterable": False}},
            "overrides": [],
        },
    }


def bars(title, targets, ds, x, y, w, h, desc=""):
    return {
        "type": "barchart",
        "title": title,
        "description": desc,
        "id": next(_id),
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": ds,
        "targets": targets,
        "options": {
            "orientation": "auto",
            "showValue": "auto",
            "xTickLabelRotation": -45,
            "legend": {"showLegend": False},
        },
        "fieldConfig": {
            "defaults": {"custom": {"fillOpacity": 80, "lineWidth": 0}},
            "overrides": [],
        },
    }


def row(title, y):
    return {
        "type": "row",
        "title": title,
        "id": next(_id),
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "panels": [],
    }


WARN = [
    {"color": "green", "value": None},
    {"color": "#EAB839", "value": 500},
    {"color": "red", "value": 5000},
]
GOOD_HIGH = [
    {"color": "red", "value": None},
    {"color": "#EAB839", "value": 30},
    {"color": "green", "value": 80},
]
AGE = [
    {"color": "green", "value": None},
    {"color": "#EAB839", "value": 3600},
    {"color": "red", "value": 43200},
]

P = []
# ── backlog ───────────────────────────────────────────────────────────────
P += [row("Backlog and drain", 0)]
P += [
    stat(
        "Load backlog",
        promq(LAG),
        PROM,
        0,
        1,
        4,
        5,
        steps=WARN,
        mode="area",
        desc="Messages written to books.clean that the catalogue has not absorbed.",
    ),
    stat(
        "Time to drain",
        promq(f"{LAG} / clamp_min({DRAIN}, 0.001)"),
        PROM,
        4,
        1,
        5,
        5,
        unit="s",
        steps=AGE,
        desc="Backlog divided by how fast it is actually falling, over ten minutes. "
        "Rises when a rebuild pauses the consumers — that is real, not noise.",
    ),
    stat(
        "Net drain",
        promq(f"{DRAIN} * 60"),
        PROM,
        9,
        1,
        4,
        5,
        unit="short",
        mode="area",
        desc="Messages per minute the backlog is shrinking by: consumed minus produced.",
    ),
    stat(
        "Produced",
        promq('sum(rate(kafka_topic_partition_current_offset{topic="books.clean"}[5m])) * 60'),
        PROM,
        13,
        1,
        4,
        5,
        desc="Messages per minute arriving on books.clean.",
    ),
    stat(
        "Consumers",
        promq('sum(kafka_consumergroup_members{consumergroup="book-pipeline-load"})'),
        PROM,
        17,
        1,
        3,
        5,
        steps=[
            {"color": "red", "value": None},
            {"color": "#EAB839", "value": 1},
            {"color": "green", "value": 3},
        ],
        desc="Three partitions means three can work. Fewer, and parallelism is idle.",
    ),
    stat(
        "Books",
        sql("catalogue", "SELECT count(*) AS value FROM books"),
        PG("catalogue"),
        20,
        1,
        4,
        5,
    ),
]
P += [
    ts(
        "Backlog over time",
        promq("sum by (consumergroup) (kafka_consumergroup_lag)", "{{consumergroup}}"),
        PROM,
        0,
        6,
        12,
        7,
        desc="Transform caught up while load lags is the normal shape: Kafka "
        "delivers faster than a database an ocean away can absorb.",
    ),
    ts(
        "Lag by partition",
        promq(
            'kafka_consumergroup_lag{consumergroup="book-pipeline-load"}', "partition {{partition}}"
        ),
        PROM,
        12,
        6,
        12,
        7,
        desc="One line well above the others means that partition's consumer is stuck or slower.",
    ),
]
# ── catalogue ─────────────────────────────────────────────────────────────
P += [row("Catalogue", 13)]
P += [
    stat(
        "Added in the last hour",
        sql(
            "catalogue",
            "SELECT count(*) AS value FROM books WHERE created_at > now() - interval '1 hour'",
        ),
        PG("catalogue"),
        0,
        14,
        4,
        5,
        mode="none",
        desc="Zero while the backlog replays books already held — re-loading one is a no-op by design.",
    ),
    stat(
        "Year coverage",
        sql(
            "catalogue",
            "SELECT round(100.0 * count(*) FILTER (WHERE published_year IS NOT NULL) / greatest(count(*),1), 1) "
            "AS value FROM books",
        ),
        PG("catalogue"),
        4,
        14,
        4,
        5,
        unit="percent",
        steps=GOOD_HIGH,
    ),
    stat(
        "ISBN coverage",
        sql(
            "catalogue",
            "SELECT round(100.0 * count(*) FILTER (WHERE isbn13 IS NOT NULL) / greatest(count(*),1), 1) "
            "AS value FROM books",
        ),
        PG("catalogue"),
        8,
        14,
        4,
        5,
        unit="percent",
        steps=GOOD_HIGH,
    ),
    ts(
        "Books added per hour",
        sql(
            "catalogue",
            "SELECT date_trunc('hour', created_at) AS time, count(*) AS books FROM books "
            "WHERE $__timeFilter(created_at) GROUP BY 1 ORDER BY 1",
            "time_series",
        ),
        PG("catalogue"),
        12,
        14,
        12,
        5,
    ),
]
P += [
    bars(
        "Books by decade",
        sql(
            "catalogue",
            "SELECT ((published_year/10)*10)::text AS decade, count(*) AS books FROM books "
            "WHERE published_year IS NOT NULL GROUP BY 1 ORDER BY 1",
        ),
        PG("catalogue"),
        0,
        19,
        12,
        7,
    ),
    table(
        "Field coverage",
        sql(
            "catalogue",
            "SELECT 'publication year' AS field, count(*) FILTER (WHERE published_year IS NOT NULL) AS present, "
            "round(100.0*count(*) FILTER (WHERE published_year IS NOT NULL)/greatest(count(*),1),1) AS pct FROM books "
            "UNION ALL SELECT 'isbn-13', count(*) FILTER (WHERE isbn13 IS NOT NULL), "
            "round(100.0*count(*) FILTER (WHERE isbn13 IS NOT NULL)/greatest(count(*),1),1) FROM books "
            "UNION ALL SELECT 'description', count(*) FILTER (WHERE description IS NOT NULL), "
            "round(100.0*count(*) FILTER (WHERE description IS NOT NULL)/greatest(count(*),1),1) FROM books "
            "UNION ALL SELECT 'publisher', count(*) FILTER (WHERE publisher IS NOT NULL), "
            "round(100.0*count(*) FILTER (WHERE publisher IS NOT NULL)/greatest(count(*),1),1) FROM books "
            "UNION ALL SELECT 'cover image', count(*) FILTER (WHERE cover_url IS NOT NULL), "
            "round(100.0*count(*) FILTER (WHERE cover_url IS NOT NULL)/greatest(count(*),1),1) FROM books",
        ),
        PG("catalogue"),
        12,
        19,
        12,
        7,
        desc="A missing field means no source supplied it, not that the value is zero.",
    ),
]
# ── provenance ────────────────────────────────────────────────────────────
P += [row("Provenance", 26)]
P += [
    table(
        "Records per source",
        sql(
            "catalogue",
            "SELECT source, count(*) AS records, count(DISTINCT book_id) AS books, "
            "max(last_seen_at) AS last_seen FROM book_sources GROUP BY source ORDER BY records DESC",
        ),
        PG("catalogue"),
        0,
        27,
        9,
        6,
    ),
    stat(
        "Goodreads enriched",
        sql(
            "catalogue",
            "SELECT round(100.0 * count(*) FILTER (WHERE raw_payload ? '_edition') / greatest(count(*),1), 1) "
            "AS value FROM book_sources WHERE source='goodreads'",
        ),
        PG("catalogue"),
        9,
        27,
        5,
        6,
        unit="percent",
        steps=GOOD_HIGH,
        desc="Share of Goodreads records carrying the work-editions page — the only place a "
        "publication year appears. Was 0% until the Accept header was fixed.",
    ),
    stat(
        "Contested, unadjudicated",
        sql(
            "catalogue",
            "SELECT count(*) AS value FROM (SELECT b.id FROM books b JOIN book_sources bs ON bs.book_id=b.id "
            "WHERE NOT EXISTS (SELECT 1 FROM book_sources g WHERE g.book_id=b.id AND g.source='goodreads') "
            "GROUP BY b.id HAVING count(DISTINCT bs.source) > 1) t",
        ),
        PG("catalogue"),
        14,
        27,
        5,
        6,
        desc="Books whose sources disagree and that the tie-breaker has never seen.",
    ),
    stat(
        "Rejected records",
        sql("catalogue", "SELECT count(*) AS value FROM rejected_records"),
        PG("catalogue"),
        19,
        27,
        5,
        6,
        steps=[{"color": "green", "value": None}, {"color": "#EAB839", "value": 100}],
    ),
]
# ── runs ──────────────────────────────────────────────────────────────────
P += [row("Runs", 33)]
P += [
    stat(
        "Oldest open run",
        sql(
            "catalogue",
            "SELECT extract(epoch FROM now() - min(started_at)) AS value FROM ingestion_runs "
            "WHERE status IN ('running','processing')",
        ),
        PG("catalogue"),
        0,
        34,
        4,
        6,
        unit="s",
        steps=AGE,
        desc="A run open for hours was killed rather than finished. Anything under an hour is "
        "usually just awaiting its consumers.",
    ),
    table(
        "Recent ingestion runs",
        sql(
            "catalogue",
            "SELECT started_at, status, records_extracted AS extracted, records_loaded AS loaded, "
            "records_rejected AS rejected FROM ingestion_runs ORDER BY started_at DESC LIMIT 10",
        ),
        PG("catalogue"),
        4,
        34,
        10,
        6,
    ),
    table(
        "Airflow dag runs",
        sql(
            "airflow",
            "SELECT start_date, dag_id, state, run_type FROM dag_run ORDER BY start_date DESC NULLS LAST LIMIT 10",
        ),
        PG("airflow"),
        14,
        34,
        10,
        6,
    ),
]

dash = {
    "uid": "pipeline-vitals",
    "title": "Pipeline Vitals",
    "description": "Backlog with a drain projection, catalogue growth, provenance and run "
    "state — the three systems that each hold part of the answer, on one screen.",
    "tags": ["catalogue", "pipeline"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 2,
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"},
    "panels": P,
}
pathlib.Path("observability/grafana/dashboards/pipeline.json").write_text(
    json.dumps(dash, indent=2)
)
print(
    "panels:",
    len([p for p in P if p["type"] != "row"]),
    "in",
    len([p for p in P if p["type"] == "row"]),
    "rows",
)
