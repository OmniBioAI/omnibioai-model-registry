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
        run_id       VARCHAR(64)  NOT NULL PRIMARY KEY,
        task         VARCHAR(128) NOT NULL,
        model_name   VARCHAR(128) NOT NULL,
        status       ENUM('running','finished','failed') NOT NULL DEFAULT 'running',
        started_at   DATETIME     NOT NULL,
        finished_at  DATETIME     NULL,
        actor        VARCHAR(128) NULL,
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
    """CREATE TABLE IF NOT EXISTS for every registry table. Safe to call on each startup."""
    with conn.cursor() as cursor:
        for ddl in _DDL:
            cursor.execute(ddl)
