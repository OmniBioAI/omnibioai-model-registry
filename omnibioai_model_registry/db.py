"""
Lightweight PyMySQL connection factory + table bootstrap for run tracking.

No ORM — plain SQL only. All functions use a caller-supplied connection so
that tracking.py functions can be tested with a connection injected in tests.
"""
from __future__ import annotations

import os

from .errors import RegistryNotConfigured

# --------------------------------------------------------------------------- #
# DDL                                                                          #
# --------------------------------------------------------------------------- #

_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS omr_runs (
        run_id            VARCHAR(64)  NOT NULL PRIMARY KEY,
        task              VARCHAR(128) NOT NULL,
        model_name        VARCHAR(128) NOT NULL,
        status            ENUM('running','finished','failed') NOT NULL DEFAULT 'running',
        started_at        DATETIME     NOT NULL,
        finished_at       DATETIME     NULL,
        actor             VARCHAR(128) NULL,
        organization_id   VARCHAR(64)  NULL,
        ownership_status  VARCHAR(20)  NOT NULL DEFAULT 'legacy_unowned',
        INDEX idx_task_model (task, model_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS omr_params (
        id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
        run_id      VARCHAR(64)  NOT NULL,
        key_name    VARCHAR(255) NOT NULL,
        value_text  TEXT         NOT NULL,
        UNIQUE KEY uq_run_key (run_id, key_name),
        FOREIGN KEY (run_id) REFERENCES omr_runs(run_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS omr_metrics (
        id          BIGINT   NOT NULL AUTO_INCREMENT PRIMARY KEY,
        run_id      VARCHAR(64)  NOT NULL,
        key_name    VARCHAR(255) NOT NULL,
        value       DOUBLE   NOT NULL,
        step        INT      NOT NULL DEFAULT 0,
        ts_utc      DATETIME NOT NULL,
        INDEX idx_run_key (run_id, key_name),
        FOREIGN KEY (run_id) REFERENCES omr_runs(run_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS omr_tags (
        id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
        run_id      VARCHAR(64)  NOT NULL,
        key_name    VARCHAR(255) NOT NULL,
        value_text  TEXT         NOT NULL,
        UNIQUE KEY uq_run_tag (run_id, key_name),
        FOREIGN KEY (run_id) REFERENCES omr_runs(run_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS omr_version_tags (
        id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
        task        VARCHAR(128) NOT NULL,
        model_name  VARCHAR(128) NOT NULL,
        version     VARCHAR(128) NOT NULL,
        key_name    VARCHAR(255) NOT NULL,
        value_text  TEXT         NOT NULL,
        UNIQUE KEY uq_ver_tag (task, model_name, version, key_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


# --------------------------------------------------------------------------- #
# Phase 2C — additive-only column migration for omr_runs                       #
# --------------------------------------------------------------------------- #
#
# `_DDL` above is `CREATE TABLE IF NOT EXISTS` -- a no-op against a table
# that already exists, so it does NOT retrofit organization_id/
# ownership_status onto an omr_runs table created by an earlier deployed
# version of this service. These two ALTER statements do that, run once
# per startup, each independently idempotent (guarded against MySQL
# error 1060 "Duplicate column name" so re-running against an
# already-migrated -- or freshly created, since _DDL already includes
# both columns -- table is always a safe no-op). Kept as two separate
# statements rather than one combined ALTER so a service crash between
# the two still self-heals cleanly on the next startup instead of
# silently skipping the second column forever.
#
# `ownership_status` defaults to 'legacy_unowned' (not 'unowned') on
# ADD COLUMN -- MySQL applies that default to every pre-existing row,
# which is exactly the Phase 2A/2B precedent for a resource nobody can
# deterministically attribute after the fact: never guessed, denied for
# every caller until a real owner logs a fresh run under that run_id (it
# can't be reclaimed -- run_id is the primary key, `INSERT IGNORE` only
# ever creates a genuinely new row). See tracking.py's
# `check_run_ownership()`/module docstring for why organization_id is
# NOT derived from ownership.json here (a run's task/model_name has no
# required correspondence to any registered model -- that's the whole
# point of experiment tracking preceding registration) and NOT backfilled
# by joining on task/model_name against ownership.json either: pre-
# Phase-2A this service enforced no uniqueness on those strings at all,
# so two different orgs' legacy runs could share the same task/model_name
# by coincidence -- a string-match backfill would risk attributing one
# org's historical run data to a different org. See PR description for
# the full reasoning.
_ALTER_DDL: list[str] = [
    "ALTER TABLE omr_runs ADD COLUMN organization_id VARCHAR(64) NULL",
    "ALTER TABLE omr_runs ADD COLUMN ownership_status VARCHAR(20) NOT NULL DEFAULT 'legacy_unowned'",
]

_MYSQL_ERRNO_DUPLICATE_COLUMN = 1060


def _run_alter_ddl_idempotent(conn) -> None:
    with conn.cursor() as cursor:
        for ddl in _ALTER_DDL:
            try:
                cursor.execute(ddl)
            except Exception as exc:
                errno = exc.args[0] if getattr(exc, "args", None) else None
                if errno == _MYSQL_ERRNO_DUPLICATE_COLUMN:
                    continue  # already migrated (or created via _DDL) -- safe to re-run
                raise


# --------------------------------------------------------------------------- #
# Connection factory                                                            #
# --------------------------------------------------------------------------- #

def get_connection():
    """
    Return a new autocommit PyMySQL connection using DB_* env vars.

    Raises RegistryNotConfigured if DB_HOST is absent so callers can return
    HTTP 503 or fall back to filesystem mode gracefully.
    """
    import pymysql
    import pymysql.cursors

    host = os.getenv("DB_HOST", "").strip()
    if not host:
        raise RegistryNotConfigured(
            "DB_HOST is not set — database not configured. "
            "Set DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME to enable "
            "DB-backed run tracking."
        )
    return pymysql.connect(
        host=host,
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "model_registry"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        connect_timeout=5,
    )


def init_tables(conn) -> None:
    """CREATE TABLE IF NOT EXISTS for every registry table, then apply the
    Phase 2C additive-only column migration (see _ALTER_DDL above) so an
    existing omr_runs table gets organization_id/ownership_status too.
    Safe to call on each startup."""
    with conn.cursor() as cursor:
        for ddl in _DDL:
            cursor.execute(ddl)
    _run_alter_ddl_idempotent(conn)
