import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import urllib.request
from typing import Any, Dict, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
REGISTRY_PATH = os.path.join(CONFIG_DIR, "model_registry.json")
STATE_PATH = os.path.join(CONFIG_DIR, "model_state.json")


DEFAULT_REGISTRY: Dict[str, Any] = {
    "profiles": {
        "text-priority": {
            "model": "diffusion-text",
            "default_version": "v1",
            "description": "Text-focused default profile",
        },
        "natural-priority": {
            "model": "diffusion-natural",
            "default_version": "v1",
            "description": "Natural-image preference profile",
        },
    },
    "models": {
        "diffusion-text": {
            "description": "Diffusion model tuned for OCR readability",
            "versions": {
                "v1": {
                    "filename": "diffusion_textzoom_bs8_latest.pth",
                    "path": "model/diffusion_textzoom_bs8_latest.pth",
                    "sha256": "",
                    "url": "",
                    "notes": "Project default baseline",
                }
            },
        },
        "diffusion-natural": {
            "description": "Diffusion model tuned for natural image quality",
            "versions": {
                "v1": {
                    "filename": "diffusion_natural_latest.pth",
                    "path": "model/diffusion_natural_latest.pth",
                    "sha256": "",
                    "url": "",
                    "notes": "Placeholder entry, update url/hash before production use",
                }
            },
        },
    },
}


DEFAULT_STATE: Dict[str, Any] = {
    "active_profile": "text-priority",
    "profile_overrides": {},
    "updated_at": None,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_config_files() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_REGISTRY, f, indent=2, ensure_ascii=False)
    if not os.path.exists(STATE_PATH):
        state = dict(DEFAULT_STATE)
        state["updated_at"] = _utc_now()
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)


def load_registry() -> Dict[str, Any]:
    ensure_config_files()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return DEFAULT_REGISTRY
    return data


def load_state() -> Dict[str, Any]:
    ensure_config_files()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return dict(DEFAULT_STATE)
    return data


def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    state = dict(state)
    state["updated_at"] = _utc_now()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def resolve_profile(registry: Dict[str, Any], state: Dict[str, Any], profile: str, version: Optional[str] = None) -> Dict[str, Any]:
    profiles = registry.get("profiles") if isinstance(registry, dict) else None
    models = registry.get("models") if isinstance(registry, dict) else None
    if not isinstance(profiles, dict) or not isinstance(models, dict):
        raise KeyError("invalid registry format")

    if profile not in profiles:
        raise KeyError(f"unknown profile: {profile}")

    profile_meta = profiles[profile]
    if not isinstance(profile_meta, dict):
        raise KeyError(f"invalid profile config: {profile}")

    model_id = str(profile_meta.get("model"))
    if model_id not in models:
        raise KeyError(f"unknown model id in profile {profile}: {model_id}")

    model_meta = models[model_id]
    versions = model_meta.get("versions") if isinstance(model_meta, dict) else None
    if not isinstance(versions, dict):
        raise KeyError(f"invalid versions config for model: {model_id}")

    override_version = None
    overrides = state.get("profile_overrides") if isinstance(state, dict) else None
    if isinstance(overrides, dict):
        v = overrides.get(profile)
        if isinstance(v, str) and v:
            override_version = v

    selected_version = version or override_version or str(profile_meta.get("default_version") or "")
    if selected_version not in versions:
        raise KeyError(f"unknown version for profile {profile}: {selected_version}")

    version_meta = versions[selected_version]
    if not isinstance(version_meta, dict):
        raise KeyError(f"invalid version metadata for {profile}:{selected_version}")

    return {
        "profile": profile,
        "model": model_id,
        "version": selected_version,
        "profile_description": profile_meta.get("description"),
        "model_description": model_meta.get("description"),
        "filename": version_meta.get("filename"),
        "path": version_meta.get("path"),
        "sha256": version_meta.get("sha256"),
        "url": version_meta.get("url"),
        "notes": version_meta.get("notes"),
    }


def compute_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model_id: str, model_path: Optional[str] = None, version: Optional[str] = None) -> Dict[str, object]:
    registry = load_registry()
    state = load_state()
    resolved = resolve_profile(registry, state, model_id, version=version)

    path = os.path.abspath(model_path or str(resolved.get("path") or ""))
    exists = os.path.exists(path)
    sha256_actual = compute_sha256(path) if exists else None
    sha256_expected = str(resolved.get("sha256") or "").strip().lower()
    hash_ok = None
    if exists and sha256_expected:
        hash_ok = (sha256_actual or "").lower() == sha256_expected

    return {
        "profile": model_id,
        "model": resolved.get("model"),
        "path": path,
        "exists": exists,
        "version": resolved.get("version"),
        "sha256_expected": sha256_expected or None,
        "sha256_actual": sha256_actual,
        "hash_ok": hash_ok,
        "url": resolved.get("url") or None,
        "checked_at": _utc_now(),
    }


def download_model(model_id: str, output_dir: str, force: bool = False, version: Optional[str] = None) -> Dict[str, object]:
    registry = load_registry()
    state = load_state()
    resolved = resolve_profile(registry, state, model_id, version=version)

    url = str(resolved.get("url") or "").strip()
    if not url:
        raise ValueError(f"model '{model_id}' has no download url configured")

    filename = str(resolved.get("filename") or f"{model_id}.pth")
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, filename)

    if os.path.exists(target) and not force:
        result = verify_model(model_id, target, version=version)
        result["downloaded"] = False
        result["reason"] = "exists"
        return result

    urllib.request.urlretrieve(url, target)
    result = verify_model(model_id, target, version=version)
    result["downloaded"] = True
    result["url"] = url
    if result.get("sha256_expected") and result.get("hash_ok") is False:
        raise RuntimeError(
            f"sha256 mismatch for {model_id}: expected={result['sha256_expected']} actual={result['sha256_actual']}"
        )
    return result


def list_models() -> Dict[str, object]:
    registry = load_registry()
    state = load_state()
    active_profile = str(state.get("active_profile") or "text-priority")

    payload = {"profiles": []}
    profiles = registry.get("profiles") if isinstance(registry, dict) else {}
    if not isinstance(profiles, dict):
        profiles = {}

    for profile in sorted(profiles.keys()):
        try:
            resolved = resolve_profile(registry, state, profile)
        except Exception as err:
            payload["profiles"].append({
                "profile": profile,
                "error": str(err),
            })
            continue

        payload["profiles"].append(
            {
                "profile": profile,
                "active": profile == active_profile,
                "model": resolved.get("model"),
                "version": resolved.get("version"),
                "path": resolved.get("path"),
                "filename": resolved.get("filename"),
                "sha256": resolved.get("sha256") or None,
                "has_url": bool(str(resolved.get("url") or "").strip()),
                "description": resolved.get("profile_description"),
                "notes": resolved.get("notes"),
            }
        )
    payload["generated_at"] = _utc_now()
    return payload


def activate_profile(profile: str, version: Optional[str] = None) -> Dict[str, object]:
    registry = load_registry()
    state = load_state()
    _ = resolve_profile(registry, state, profile, version=version)

    state["active_profile"] = profile
    overrides = state.get("profile_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    if version:
        overrides[profile] = version
    state["profile_overrides"] = overrides
    save_state(state)

    resolved = resolve_profile(registry, state, profile)
    return {
        "active_profile": profile,
        "model": resolved.get("model"),
        "version": resolved.get("version"),
        "path": resolved.get("path"),
        "updated_at": _utc_now(),
    }


def model_status() -> Dict[str, object]:
    state = load_state()
    active_profile = str(state.get("active_profile") or "text-priority")
    info = verify_model(active_profile)
    info["active_profile"] = active_profile
    info["state_file"] = os.path.abspath(STATE_PATH)
    info["registry_file"] = os.path.abspath(REGISTRY_PATH)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Model registry utility: list/verify/download/activate/status")
    parser.add_argument("--action", choices=["list", "verify", "download", "activate", "status"], required=True)
    parser.add_argument("--model", type=str, default=None, help="Model id")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path for verify")
    parser.add_argument("--output_dir", type=str, default="model", help="Output directory for download")
    parser.add_argument("--version", type=str, default=None, help="Model version override")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    if args.action == "list":
        print(json.dumps(list_models(), indent=2, ensure_ascii=False))
        return

    if args.action == "status":
        print(json.dumps(model_status(), indent=2, ensure_ascii=False))
        return

    if not args.model:
        raise ValueError("--model is required for verify/download/activate")

    if args.action == "activate":
        print(json.dumps(activate_profile(args.model, version=args.version), indent=2, ensure_ascii=False))
        return

    if args.action == "verify":
        print(json.dumps(verify_model(args.model, args.model_path, version=args.version), indent=2, ensure_ascii=False))
        return

    print(json.dumps(download_model(args.model, args.output_dir, force=args.force, version=args.version), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
