"""Composizione YAML senza override impliciti e snapshot riproducibili."""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

PROFILE_NAMES = {"preprocessing", "features", "filters", "advanced"}


class UniqueKeyLoader(yaml.SafeLoader):
    """Rifiuta anche le chiavi duplicate nello stesso documento YAML."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Le chiavi YAML devono essere stringhe")
            if key in result:
                raise ValueError(f"Chiave YAML duplicata '{key}' alla riga {key_node.start_mark.line + 1}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def read_mapping(path: Path) -> tuple[dict, bytes]:
    content = path.read_bytes()
    try:
        data = yaml.load(content.decode("utf-8-sig"), Loader=UniqueKeyLoader)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Configurazione non valida in '{path}': {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"'{path}' deve contenere una mappa YAML, non un valore o una lista")
    return data, content


def load_profiles(path: Path) -> tuple[dict, dict[str, bytes]]:
    path = path.resolve()
    data, content = read_mapping(path)
    sources = {str(path): content}
    profiles = data.pop("profiles", {})
    if not isinstance(profiles, dict) or set(profiles) - PROFILE_NAMES:
        raise ValueError("profiles deve essere una mappa con chiavi preprocessing, features, filters, advanced")
    mirp = data.setdefault("mirp", {})
    if not isinstance(mirp, dict):
        raise ValueError("mirp deve essere una mappa YAML")
    owners = {key: str(path) for key in mirp}
    for name, relative_path in profiles.items():
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"profiles.{name} deve indicare il percorso di un file YAML")
        profile_path = (path.parent / Path(relative_path).expanduser()).resolve()
        if str(profile_path) in sources:
            raise ValueError(f"Profilo caricato piu' volte o riferimento circolare: '{profile_path}'")
        profile, profile_content = read_mapping(profile_path)
        if set(profile) != {"mirp"} or not isinstance(profile["mirp"], dict):
            raise ValueError(f"Il profilo '{profile_path}' deve contenere solo la sezione mirp; niente profili annidati")
        for key, value in profile["mirp"].items():
            if key in owners:
                raise ValueError(f"Parametro mirp.{key} duplicato in '{owners[key]}' e '{profile_path}'")
            owners[key] = str(profile_path)
            mirp[key] = value
        sources[str(profile_path)] = profile_content
    return data, sources


def yaml_value(value: Any) -> Any:
    """Conserva NaN come .nan e converte oggetti MIRP e Path in YAML sicuro."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): yaml_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [yaml_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return yaml_value(vars(value))
    return value


def save_configuration_snapshot(config, output_dir: Path, *, mirp_config=None) -> Path:
    """Salva i byte letti all'avvio, non file eventualmente cambiati durante l'attesa."""
    from mirp.settings.generic import SettingsClass

    active = mirp_config or config.mirp
    resolved = yaml_value(config.model_dump(exclude={"provenance"}))
    resolved["mirp"] = yaml_value(active.model_dump())
    original_dir = output_dir / "config_sources"
    original_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (source, content) in enumerate(config._source_files.items()):
        name = f"{index:02d}_{Path(source).name}"
        (original_dir / name).write_bytes(content)
        manifest.append({"source": source, "copy": f"config_sources/{name}", "sha256": sha256(content).hexdigest()})
    resolved["provenance"] = {
        "versions": {name: version(name) for name in ("mirp", "numpy", "pydantic", "PyYAML")},
        "sources": manifest,
        "effective_mirp_settings": {
            f"{spacing[0]:g}mm": yaml_value(SettingsClass(**active.for_voxel_spacing(spacing).to_mirp_kwargs()))
            for spacing in active.voxel_spacings()
        },
    }
    text = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    target = output_dir / "config_resolved.yaml"
    target.write_text(text, encoding="utf-8")
    return target
