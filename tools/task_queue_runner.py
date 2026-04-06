import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import run_all


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_queue(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        tasks = data.get("tasks", [])
    elif isinstance(data, list):
        tasks = data
    else:
        tasks = []
    return [t for t in tasks if isinstance(t, dict)]


def save_history(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_argv(task: Dict[str, Any]) -> List[str]:
    if isinstance(task.get("argv"), list):
        return [str(x) for x in task["argv"]]

    command = str(task.get("command", "")).strip()
    args_obj = task.get("args", {})
    if not command:
        return []

    argv = [command]
    if isinstance(args_obj, dict):
        for k, v in args_obj.items():
            key = f"--{k}"
            if isinstance(v, bool):
                if v:
                    argv.append(key)
            elif isinstance(v, list):
                for item in v:
                    argv.extend([key, str(item)])
            elif v is not None:
                argv.extend([key, str(v)])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queued run_all tasks and persist task history")
    parser.add_argument("--queue_json", type=str, required=True, help="Queue definition json")
    parser.add_argument("--history_json", type=str, default="queue_history.json", help="History output json")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop queue when a task fails")
    args = parser.parse_args()

    queue_path = os.path.abspath(args.queue_json)
    if not os.path.exists(queue_path):
        raise FileNotFoundError(f"queue_json not found: {queue_path}")

    tasks = load_queue(queue_path)
    history: Dict[str, Any] = {
        "queue_json": queue_path,
        "started_at": _utc_now(),
        "finished_at": None,
        "stop_on_error": bool(args.stop_on_error),
        "tasks": [],
    }

    for idx, task in enumerate(tasks, start=1):
        argv = normalize_argv(task)
        name = str(task.get("name") or f"task-{idx}")
        item = {
            "index": idx,
            "name": name,
            "argv": argv,
            "started_at": _utc_now(),
            "finished_at": None,
            "return_code": None,
            "status": "running",
        }

        if not argv:
            item["status"] = "invalid"
            item["return_code"] = 2
            item["finished_at"] = _utc_now()
            history["tasks"].append(item)
            if args.stop_on_error:
                break
            continue

        try:
            rc = int(run_all.main(argv))
        except SystemExit as ex:
            rc = int(ex.code) if isinstance(ex.code, int) else 1
        except Exception:
            rc = 1

        item["return_code"] = rc
        item["status"] = "success" if rc == 0 else "failed"
        item["finished_at"] = _utc_now()
        history["tasks"].append(item)

        if rc != 0 and args.stop_on_error:
            break

    history["finished_at"] = _utc_now()
    history["success_count"] = sum(1 for t in history["tasks"] if t.get("status") == "success")
    history["failed_count"] = sum(1 for t in history["tasks"] if t.get("status") == "failed")
    history["invalid_count"] = sum(1 for t in history["tasks"] if t.get("status") == "invalid")

    save_history(args.history_json, history)
    print(f"Queue history saved: {os.path.abspath(args.history_json)}")
    print(json.dumps(history, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
