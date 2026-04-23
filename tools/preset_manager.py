import argparse
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
PRESET_PATH = os.path.join(CONFIG_DIR, "custom_presets.json")


DEFAULT_PRESET_FILE: Dict[str, Any] = {
    "version": 1,
    "updated_at": None,
    "presets": {},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_preset_file() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(PRESET_PATH):
        payload = dict(DEFAULT_PRESET_FILE)
        payload["updated_at"] = _utc_now()
        with open(PRESET_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def load_presets() -> Dict[str, Any]:
    ensure_preset_file()
    with open(PRESET_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        payload = dict(DEFAULT_PRESET_FILE)
    presets = payload.get("presets")
    if not isinstance(presets, dict):
        payload["presets"] = {}
    return payload


def save_presets(payload: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = _utc_now()
    with open(PRESET_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def list_presets() -> Dict[str, Any]:
    payload = load_presets()
    payload["preset_count"] = len(payload.get("presets", {}))
    return payload


def upsert_preset(name: str, values: Dict[str, Any], description: str = "") -> Dict[str, Any]:
    payload = load_presets()
    presets = payload.setdefault("presets", {})
    if not isinstance(presets, dict):
        presets = {}
        payload["presets"] = presets

    presets[name] = {
        "values": values,
        "description": description,
        "updated_at": _utc_now(),
    }
    save_presets(payload)
    return {
        "name": name,
        "values": values,
        "description": description,
        "saved": True,
    }


def delete_preset(name: str) -> Dict[str, Any]:
    payload = load_presets()
    presets = payload.get("presets", {})
    removed = False
    if isinstance(presets, dict) and name in presets:
        del presets[name]
        removed = True
        save_presets(payload)
    return {
        "name": name,
        "removed": removed,
    }


def parse_values(raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as err:
        raise ValueError(f"invalid values json: {err}")
    if not isinstance(obj, dict):
        raise ValueError("values must be a JSON object")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom preset manager")
    parser.add_argument("--action", choices=["list", "set", "delete"], required=True)
    parser.add_argument("--name", type=str, default=None, help="Preset name")
    parser.add_argument("--values_json", type=str, default=None, help="Preset values as JSON object")
    parser.add_argument("--description", type=str, default="", help="Preset description")
    args = parser.parse_args()

    if args.action == "list":
        print(json.dumps(list_presets(), indent=2, ensure_ascii=False))
        return

    if not args.name:
        raise ValueError("--name is required for set/delete")

    if args.action == "delete":
        print(json.dumps(delete_preset(args.name), indent=2, ensure_ascii=False))
        return

    if not args.values_json:
        raise ValueError("--values_json is required for set")

    values = parse_values(args.values_json)
    print(json.dumps(upsert_preset(args.name, values, description=args.description), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
