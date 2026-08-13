# File: omnibioai_model_registry/cli/main.py
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from omnibioai_model_registry import (
    ModelRegistry,
    promote_model,
    register_model,
    resolve_model,
    verify_model_ref,
)
from omnibioai_model_registry.errors import InvalidStageTransition, ModelRegistryError


def cmd_list(args):
    from omnibioai_model_registry.package import layout as L

    registry = ModelRegistry.from_env()
    root = registry.root

    # Uses layout.py's task_root() (validated, see path_safety.py)
    # rather than a raw join, consistent with every other path this
    # service constructs -- not a meaningfully different trust boundary
    # for a local CLI operator, but keeps a single safe primitive
    # everywhere rather than a hand-rolled exception here.
    models_dir = L.task_root(root, args.task) / "models"
    if not models_dir.exists():
        print(f"No models found for task '{args.task}'")
        return

    for model_dir in sorted(models_dir.iterdir()):
        if model_dir.is_dir():
            print(model_dir.name)


def cmd_resolve(args):
    path = resolve_model(args.task, args.model_ref)
    print(path)


def cmd_promote(args):
    promote_model(
        task=args.task,
        model_name=args.model,
        alias=args.alias,
        version=args.version,
        actor=args.actor,
        reason=args.reason,
    )
    print(f"Promoted {args.model}@{args.version} → {args.alias}")


def cmd_verify(args):
    verify_model_ref(args.task, args.model_ref)
    print("Integrity verification passed.")


def _read_json_file(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text())


def cmd_register(args):
    meta = {}
    if args.metadata_json:
        meta.update(_read_json_file(args.metadata_json))
    if args.metadata_inline:
        meta.update(json.loads(args.metadata_inline))

    out = register_model(
        task=args.task,
        model_name=args.model,
        version=args.version,
        artifacts_dir=args.artifacts,
        metadata=meta,
        set_alias=args.set_alias,
        actor=args.actor,
        reason=args.reason,
        # Phase 2A: the CLI is an out-of-band, operator-run trusted tool
        # with no IAM/JWT identity of its own (unlike the HTTP API, which
        # always derives this from a verified token) -- --org-id lets an
        # administrator explicitly assign ownership for a brand-new model
        # at registration time. It has no effect on an already-owned or
        # legacy model (ownership is write-once; see ownership.py).
        organization_id=args.org_id,
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Registered: {out['task']}/{out['model_name']}/{out['version']}")
        print(f"Path: {out['package_path']}")
        if out.get("alias_set"):
            print(f"Alias set: {out['alias_set']}")
        print(f"Ownership: organization_id={out.get('organization_id')} status={out.get('ownership_status')}")


def cmd_migrate_ownership(args):
    from omnibioai_model_registry.ownership import backfill_legacy_ownership

    registry = ModelRegistry.from_env()
    summary = backfill_legacy_ownership(registry.root)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Scanned:               {summary['scanned']}")
        print(f"Migrated to legacy:    {summary['migrated']}")
        print(f"Already had ownership: {summary['already_had_ownership']}")
        if summary["migrated"]:
            print(
                "\nMigrated models are recorded as status=legacy_unowned "
                "(organization_id=null). They require manual administrator "
                "assignment in a later phase -- this command never guesses "
                "an organization from actor strings or other weak evidence."
            )


def cmd_show(args):
    registry = ModelRegistry.from_env()
    vdir = registry.resolve_model(
        task=args.task, model_ref=args.model_ref, verify=args.verify
    )

    meta_path = vdir / "model_meta.json"
    if not meta_path.exists():
        print(f"model_meta.json not found in: {vdir}", file=sys.stderr)
        sys.exit(1)

    txt = meta_path.read_text()
    meta = json.loads(txt)

    if args.raw:
        print(txt, end="" if txt.endswith("\n") else "\n")
        return

    if args.json:
        print(json.dumps(meta, indent=2))
        return

    # Pretty minimal human view
    print(f"Task:       {meta.get('task', args.task)}")
    print(f"Model:      {meta.get('model_name')}")
    print(f"Version:    {meta.get('version')}")
    print(f"Created:    {meta.get('created_at')}")
    print(f"Framework:  {meta.get('framework')}")
    print(f"Model type: {meta.get('model_type')}")

    prov = meta.get("provenance") or {}
    if prov:
        print("Provenance:")
        if prov.get("git_commit"):
            print(f"  git_commit: {prov.get('git_commit')}")
        if prov.get("training_data_ref"):
            print(f"  training_data_ref: {prov.get('training_data_ref')}")
        if prov.get("trainer_version"):
            print(f"  trainer_version: {prov.get('trainer_version')}")

    print(f"Package dir: {vdir}")


# ── Phase-3 helpers ───────────────────────────────────────────────────────────

def _resolve_vdir_no_verify(registry: ModelRegistry, task: str, model_ref: str) -> Path:
    """Resolve model ref → version dir without triggering Cython integrity checks."""
    from omnibioai_model_registry.refs import parse_model_ref
    from omnibioai_model_registry.package import layout as L
    from omnibioai_model_registry.errors import ModelNotFound

    ref = parse_model_ref(model_ref)
    alias_file = L.alias_path(registry.root, task, ref.model_name, ref.selector)
    if alias_file.exists():
        version = json.loads(alias_file.read_text())["version"]
    else:
        version = ref.selector

    vdir = L.version_dir(registry.root, task, ref.model_name, version)
    if not vdir.exists():
        raise ModelNotFound(f"Model not found: task={task}, ref={model_ref}")
    return vdir


# ── Phase-3 commands ──────────────────────────────────────────────────────────

def cmd_metrics(args):
    registry = ModelRegistry.from_env()
    vdir = _resolve_vdir_no_verify(registry, args.task, args.ref)

    metrics_path = vdir / "metrics.json"
    version_metrics: dict = {}
    if metrics_path.exists():
        try:
            version_metrics = json.loads(metrics_path.read_text())
        except Exception:
            pass

    if args.json:
        print(json.dumps(version_metrics, indent=2))
        return

    if version_metrics:
        print("Version metrics:")
        for k, v in version_metrics.items():
            print(f"  {k}: {v}")
    else:
        print("No version metrics found.")

    # Optionally show step history from DB
    meta_path = vdir / "model_meta.json"
    run_id = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            run_id = (meta.get("lineage") or {}).get("run_id")
        except Exception:
            pass

    if run_id:
        try:
            from omnibioai_model_registry import db, tracking
            conn = db.get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT key_name FROM omr_metrics WHERE run_id = %s",
                        (run_id,),
                    )
                    keys = [r["key_name"] for r in cur.fetchall()]
                if keys:
                    print(f"\nRun history (run_id={run_id}):")
                    for key in keys:
                        history = tracking.get_metric_history(conn, run_id, key)
                        values = [h["value"] for h in history]
                        print(f"  {key}: {values}")
            finally:
                conn.close()
        except Exception as exc:
            print(f"  [DB unavailable — step history skipped: {exc}]", file=sys.stderr)


def cmd_aliases(args):
    from omnibioai_model_registry.package.layout import aliases_root

    registry = ModelRegistry.from_env()
    aliases_dir = aliases_root(registry.root, args.task, args.model)

    entries = []
    if aliases_dir.exists():
        for f in sorted(aliases_dir.glob("*.json")):
            try:
                entries.append(json.loads(f.read_text()))
            except Exception:
                continue

    if args.json:
        print(json.dumps(entries, indent=2))
        return

    if not entries:
        print("No aliases found.")
        return

    col = 22
    hdr = f"{'ALIAS':<{col}} {'VERSION':<{col}} {'UPDATED_AT':<28} {'ACTOR':<{col}}"
    print(hdr)
    print("-" * len(hdr))
    for e in entries:
        print(
            f"{str(e.get('alias', '?')):<{col}} "
            f"{str(e.get('version', '?')):<{col}} "
            f"{str(e.get('updated_at', '?')):<28} "
            f"{str(e.get('actor') or '-'):<{col}}"
        )


def cmd_tag(args):
    registry = ModelRegistry.from_env()
    vdir = _resolve_vdir_no_verify(registry, args.task, args.ref)

    meta_path = vdir / "model_meta.json"
    if not meta_path.exists():
        print(f"model_meta.json not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    tags = meta.get("tags") or {}
    tags[args.key] = args.value
    meta["tags"] = tags
    text = json.dumps(meta, indent=2) + "\n"

    fd, tmp = tempfile.mkstemp(dir=meta_path.parent, suffix=".tmp")
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.replace(tmp, meta_path)

    # Persist to DB if available
    try:
        from omnibioai_model_registry import db, tracking
        conn = db.get_connection()
        version = meta.get("version", vdir.name)
        model_name = meta.get("model_name", "")
        try:
            tracking.set_version_tag(conn, args.task, model_name, version, args.key, args.value)
        finally:
            conn.close()
    except Exception:
        pass

    print(f"Tagged {args.ref} with {args.key}={args.value}")


_VALID_STAGES = frozenset({"none", "staging", "production", "archived"})


def cmd_stage(args):
    if args.stage not in _VALID_STAGES:
        raise InvalidStageTransition(
            f"Invalid stage '{args.stage}'. Must be one of: {sorted(_VALID_STAGES)}"
        )

    from omnibioai_model_registry.package.layout import version_dir as vdir_fn
    from omnibioai_model_registry.errors import ModelNotFound

    registry = ModelRegistry.from_env()
    vdir = vdir_fn(registry.root, args.task, args.model, args.version)
    if not vdir.exists():
        raise ModelNotFound(
            f"Version not found: {args.task}/{args.model}/{args.version}"
        )

    meta_path = vdir / "model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["stage"] = args.stage
        text = json.dumps(meta, indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(dir=meta_path.parent, suffix=".tmp")
        try:
            os.write(fd, text.encode())
        finally:
            os.close(fd)
        os.replace(tmp, meta_path)

    if args.stage in ("staging", "production"):
        promote_model(
            task=args.task,
            model_name=args.model,
            alias=args.stage,
            version=args.version,
            actor=args.actor,
            reason=args.reason or f"stage transition to {args.stage}",
        )

    print(f"Set {args.model}@{args.version} to stage={args.stage}")


def cmd_compare(args):
    from omnibioai_model_registry.package.layout import version_dir as vdir_fn

    registry = ModelRegistry.from_env()
    all_metrics: dict = {}
    for ver in args.versions:
        vdir = vdir_fn(registry.root, args.task, args.model, ver)
        m: dict = {}
        if (vdir / "metrics.json").exists():
            try:
                m = json.loads((vdir / "metrics.json").read_text())
            except Exception:
                pass
        all_metrics[ver] = m

    if args.json:
        result = {"versions": {v: {"metrics": all_metrics[v]} for v in args.versions}}
        print(json.dumps(result, indent=2))
        return

    all_keys = sorted({k for m in all_metrics.values() for k in m})
    if not all_keys:
        print("No metrics found for any version.")
        return

    col = 16
    header = f"{'METRIC':<26}" + "".join(f"{v:<{col}}" for v in args.versions)
    print(header)
    print("-" * len(header))

    for key in all_keys:
        vals = {v: all_metrics[v].get(key) for v in args.versions}
        numeric = {v: val for v, val in vals.items() if isinstance(val, (int, float))}
        best = max(numeric, key=lambda v: numeric[v]) if numeric else None
        row = f"{key:<26}"
        for ver in args.versions:
            val = vals.get(ver)
            if val is None:
                cell = "-"
            elif isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            if ver == best and len(args.versions) > 1:
                cell += "*"
            row += f"{cell:<{col}}"
        print(row)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="omr",
        description="OmniBioAI Model Registry CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List models for a task")
    p_list.add_argument("--task", required=True, help="Task name")
    p_list.set_defaults(func=cmd_list)

    # resolve
    p_resolve = subparsers.add_parser(
        "resolve", help="Resolve model reference to a local path"
    )
    p_resolve.add_argument("--task", required=True, help="Task name")
    p_resolve.add_argument(
        "--ref",
        dest="model_ref",
        required=True,
        help="Model reference (model@alias_or_version)",
    )
    p_resolve.set_defaults(func=cmd_resolve)

    # show
    p_show = subparsers.add_parser("show", help="Show model metadata (model_meta.json)")
    p_show.add_argument("--task", required=True, help="Task name")
    p_show.add_argument(
        "--ref",
        dest="model_ref",
        required=True,
        help="Model reference (model@alias_or_version)",
    )
    p_show.add_argument(
        "--verify", action="store_true", help="Verify package integrity before showing"
    )
    p_show.add_argument(
        "--json", action="store_true", help="Print formatted JSON metadata"
    )
    p_show.add_argument("--raw", action="store_true", help="Print raw model_meta.json")
    p_show.set_defaults(func=cmd_show)

    # register
    p_register = subparsers.add_parser(
        "register", help="Register a model package into the registry"
    )
    p_register.add_argument("--task", required=True, help="Task name")
    p_register.add_argument("--model", required=True, help="Model name")
    p_register.add_argument(
        "--version", required=True, help="Version string (immutable)"
    )
    p_register.add_argument(
        "--artifacts",
        required=True,
        help="Directory containing model package artifacts",
    )
    p_register.add_argument(
        "--set-alias",
        dest="set_alias",
        default="latest",
        help="Alias to set after register (default: latest). Use '' to skip.",
    )
    p_register.add_argument("--actor", default=None, help="Actor registering the model")
    p_register.add_argument(
        "--reason", default="cli register", help="Reason for registration"
    )
    p_register.add_argument(
        "--metadata-json", default=None, help="Path to JSON file with metadata to merge"
    )
    p_register.add_argument(
        "--metadata-inline", default=None, help="Inline JSON string metadata to merge"
    )
    p_register.add_argument(
        "--org-id",
        dest="org_id",
        default=None,
        help=(
            "Organization ID to assign as owner if this is a brand-new "
            "model (Phase 2A; operator/admin use only -- no effect on an "
            "already-owned or legacy model)"
        ),
    )
    p_register.add_argument("--json", action="store_true", help="Print JSON result")
    p_register.set_defaults(func=cmd_register)

    # promote
    p_promote = subparsers.add_parser(
        "promote", help="Promote model version to an alias"
    )
    p_promote.add_argument("--task", required=True, help="Task name")
    p_promote.add_argument("--model", required=True, help="Model name")
    p_promote.add_argument("--version", required=True, help="Version string")
    p_promote.add_argument(
        "--alias", required=True, help="Alias name (e.g. production)"
    )
    p_promote.add_argument("--actor", default=None, help="Actor performing promotion")
    p_promote.add_argument("--reason", default=None, help="Reason for promotion")
    p_promote.set_defaults(func=cmd_promote)

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify model integrity")
    p_verify.add_argument("--task", required=True, help="Task name")
    p_verify.add_argument(
        "--ref", dest="model_ref", required=True, help="Model reference"
    )
    p_verify.set_defaults(func=cmd_verify)

    # metrics
    p_metrics = subparsers.add_parser(
        "metrics", help="Show version metrics and run history"
    )
    p_metrics.add_argument("--task", required=True, help="Task name")
    p_metrics.add_argument("--ref", required=True, help="Model reference (model@version_or_alias)")
    p_metrics.add_argument("--json", action="store_true", help="Print raw JSON")
    p_metrics.set_defaults(func=cmd_metrics)

    # aliases
    p_aliases = subparsers.add_parser("aliases", help="List aliases for a model")
    p_aliases.add_argument("--task", required=True, help="Task name")
    p_aliases.add_argument("--model", required=True, help="Model name")
    p_aliases.add_argument("--json", action="store_true", help="Print raw JSON")
    p_aliases.set_defaults(func=cmd_aliases)

    # tag
    p_tag = subparsers.add_parser("tag", help="Set a tag on a model version")
    p_tag.add_argument("--task", required=True, help="Task name")
    p_tag.add_argument("--ref", required=True, help="Model reference (model@version_or_alias)")
    p_tag.add_argument("--key", required=True, help="Tag key")
    p_tag.add_argument("--value", required=True, help="Tag value")
    p_tag.add_argument("--actor", default=None, help="Actor name")
    p_tag.set_defaults(func=cmd_tag)

    # stage
    p_stage = subparsers.add_parser(
        "stage", help="Set the lifecycle stage of a model version"
    )
    p_stage.add_argument("--task", required=True, help="Task name")
    p_stage.add_argument("--model", required=True, help="Model name")
    p_stage.add_argument("--version", required=True, help="Version string")
    p_stage.add_argument(
        "--stage", required=True,
        help="Stage: none | staging | production | archived",
    )
    p_stage.add_argument("--actor", default=None, help="Actor performing the transition")
    p_stage.add_argument("--reason", default=None, help="Reason for the transition")
    p_stage.set_defaults(func=cmd_stage)

    # compare
    p_compare = subparsers.add_parser(
        "compare", help="Compare metrics across model versions"
    )
    p_compare.add_argument("--task", required=True, help="Task name")
    p_compare.add_argument("--model", required=True, help="Model name")
    p_compare.add_argument(
        "--versions", nargs="+", required=True, help="Two or more version strings"
    )
    p_compare.add_argument("--json", action="store_true", help="Print raw JSON")
    p_compare.set_defaults(func=cmd_compare)

    # migrate-ownership (Phase 2A)
    p_migrate = subparsers.add_parser(
        "migrate-ownership",
        help=(
            "Backfill an explicit legacy_unowned ownership record for every "
            "pre-existing model that has none. Deterministic, repeatable, "
            "additive-only -- never guesses an organization."
        ),
    )
    p_migrate.add_argument("--json", action="store_true", help="Print JSON summary")
    p_migrate.set_defaults(func=cmd_migrate_ownership)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle special case: user wants no alias set
    if getattr(args, "set_alias", None) == "":
        args.set_alias = None

    try:
        args.func(args)
    except ModelRegistryError as e:
        print(f"[Registry Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Unexpected Error] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
