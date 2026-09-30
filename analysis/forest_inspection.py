"""Ispezione offline delle foreste salvate; nessun fit e nessuna modifica ai modelli."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from html import escape
from pathlib import Path
from textwrap import wrap

import joblib
import numpy as np
import pandas as pd
from prompt_toolkit.shortcuts import choice
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .selector import SELECTOR_STYLE
from .tuning_config import load_tuning_configuration


def discover_forest_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Cartella tuning non trovata: {root}")
    runs = [p for p in root.iterdir() if (p / "models/random_forest.joblib").is_file()]
    runs.sort(key=lambda p: ((p / "models/random_forest.joblib").stat().st_mtime_ns, p.name), reverse=True)
    if not runs:
        raise FileNotFoundError(f"Nessuna Random Forest salvata in: {root}")
    return runs


def feature_coordinates(pipeline: Pipeline) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Ricava nomi e conversione x_originale = x_modello * scale + offset."""
    count = pipeline.n_features_in_
    names = np.asarray(getattr(pipeline, "feature_names_in_", [f"feature_{i}" for i in range(count)]), dtype=object)
    scale, offset = np.ones(count), np.zeros(count)
    for name, step in pipeline.steps[:-1]:
        if step is None or (isinstance(step, str) and step == "passthrough"):
            continue
        if isinstance(step, SimpleImputer):
            if step.add_indicator or not step.keep_empty_features:
                raise ValueError("Ispezione richiede un imputer senza indicatori e con keep_empty_features=True")
        elif isinstance(step, (StandardScaler, RobustScaler)):
            center = (
                step.mean_ if step.with_mean else None
            ) if isinstance(step, StandardScaler) else step.center_
            factor = step.scale_
            if center is not None:
                offset += scale * center
            if factor is not None:
                scale *= factor
        elif hasattr(step, "get_support"):
            support = np.asarray(step.get_support(), dtype=bool)
            if len(support) != len(names):
                raise ValueError(f"Mappa delle feature incompatibile nello step {name}")
            names, scale, offset = names[support], scale[support], offset[support]
        else:
            raise ValueError(f"Trasformazione non supportata per riconvertire le soglie: {name}")
    if len(names) != pipeline.named_steps["model"].n_features_in_:
        raise ValueError("Numero di nomi delle feature diverso da quello atteso dalla foresta")
    return [str(name) for name in names], scale, offset


def _tree_svg(estimator, names, scale, offset, classes, max_depth: int) -> str:
    tree = estimator.tree_
    positions, visible = {}, []
    leaf_number = 0

    def layout(node, depth):
        nonlocal leaf_number
        visible.append((node, depth))
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1 or depth >= max_depth:
            x = 170 + leaf_number * 340
            leaf_number += 1
        else:
            x = (layout(left, depth + 1) + layout(right, depth + 1)) / 2
        positions[node] = (x, 30 + depth * 225)
        return x

    layout(0, 0)
    height = 225 * (max(depth for _, depth in visible) + 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{max(340, leaf_number * 340)}" height="{height}" role="img" aria-label="Albero decisionale">', '<rect width="100%" height="100%" fill="#f5f7fb"/>']
    for node, depth in visible:
        x, y = positions[node]
        for child, label in ((tree.children_left[node], "si / <="), (tree.children_right[node], "no / >")):
            if child not in positions:
                continue
            cx, cy = positions[child]
            parts.append(f'<path d="M{x},{y+180} L{cx},{cy}" stroke="#667085" fill="none"/>')
            parts.append(f'<text x="{(x+cx)/2}" y="{(y+180+cy)/2}" font-size="12">{escape(label)}</text>')
    for node, depth in visible:
        x, y = positions[node]
        feature = tree.feature[node]
        leaf = tree.children_left[node] == -1
        values = np.asarray(tree.value[node]).reshape(-1)
        predicted = str(classes[int(np.argmax(values))])
        lines = [f"Nodo {node} | profondita' {depth}"]
        if not leaf:
            lines.extend(wrap(names[feature], width=38)[:2])
            threshold = tree.threshold[node]
            lines.append(f"Soglia originale <= {threshold * scale[feature] + offset[feature]:.7g}")
            lines.append(f"Soglia modello <= {threshold:.7g}")
        else:
            lines.append("FOGLIA")
        lines.extend([
            f"Campioni: {tree.n_node_samples[node]} | impurita': {tree.impurity[node]:.4g}",
            f"Classe prevalente: {predicted}",
        ])
        if not leaf and depth >= max_depth:
            lines.append("... sottoalbero non visualizzato")
        fill = "#dcfce7" if leaf else "#e0edff"
        parts.append(f'<g><title>{escape(names[feature] if not leaf else "Foglia")}</title><rect x="{x-158}" y="{y}" width="316" height="180" rx="9" fill="{fill}" stroke="#98a2b3"/>')
        for index, line in enumerate(lines):
            parts.append(f'<text x="{x-148}" y="{y+23+index*20}" font-family="monospace" font-size="12">{escape(line)}</text>')
        parts.append('</g>')
    parts.append('</svg>')
    return "\n".join(parts)


def inspect_forest(run_dir: Path, *, output_dir: Path | None = None, max_depth: int = 4) -> Path:
    if not 0 <= max_depth <= 10:
        raise ValueError("max_depth deve essere tra 0 e 10; il CSV dei nodi resta sempre completo")
    run_dir = run_dir.expanduser().resolve()
    model_path = run_dir / "models/random_forest.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Random Forest non trovata: {model_path}")
    # Joblib/pickle puo' eseguire codice: caricare esclusivamente artefatti fidati.
    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        raise ValueError(
            f"Impossibile caricare il modello {model_path}: {error}. "
            "Usa l'ambiente Python e le versioni del training."
        ) from error
    if not isinstance(pipeline, Pipeline) or not isinstance(pipeline.named_steps.get("model"), RandomForestClassifier):
        raise ValueError("Il file non contiene una pipeline RandomForestClassifier supportata")
    forest = pipeline.named_steps["model"]
    check_is_fitted(forest)
    if forest.n_outputs_ != 1:
        raise ValueError("Le foreste multi-output non sono supportate")
    names, scale, offset = feature_coordinates(pipeline)
    root = (output_dir or run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = f"rf-inspection-{datetime.now():%Y-%m-%d_%H-%M-%S}"
    for counter in range(1000):
        report = root / (base if counter == 0 else f"{base}_{counter+1}")
        try:
            report.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Impossibile creare una nuova cartella di ispezione")
    (report / "trees").mkdir()
    rows, nodes = [], []
    for index, estimator in enumerate(forest.estimators_, start=1):
        tree = estimator.tree_
        rows.append({"tree": index, "depth": estimator.get_depth(), "nodes": tree.node_count, "leaves": estimator.get_n_leaves()})
        (report / "trees" / f"tree-{index:04d}.svg").write_text(
            _tree_svg(estimator, names, scale, offset, forest.classes_, max_depth), encoding="utf-8",
        )
        for node in range(tree.node_count):
            feature = tree.feature[node]
            leaf = tree.children_left[node] == -1
            nodes.append({
                "tree": index, "node": node, "left": int(tree.children_left[node]), "right": int(tree.children_right[node]),
                "feature": None if leaf else names[feature],
                "threshold_model": None if leaf else float(tree.threshold[node]),
                "threshold_original": None if leaf else float(tree.threshold[node] * scale[feature] + offset[feature]),
                "samples": int(tree.n_node_samples[node]), "impurity": float(tree.impurity[node]),
                "predicted_class": str(forest.classes_[int(np.argmax(tree.value[node]))]),
            })
    stats = pd.DataFrame(rows)
    stats.to_csv(report / "trees.csv", sep=";", index=False)
    pd.DataFrame(nodes).to_csv(report / "nodes.csv", sep=";", index=False)
    importance = pd.DataFrame({"feature": names, "importance": forest.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(report / "feature_importances.csv", sep=";", index=False)
    summary = {
        "model_path": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "trees": len(rows), "features": len(names), "configured_max_depth": forest.max_depth,
        "depth_min": int(stats.depth.min()), "depth_mean": float(stats.depth.mean()), "depth_max": int(stats.depth.max()),
        "display_max_depth": max_depth,
        "scaler": type(pipeline.named_steps["scaler"]).__name__ if "scaler" in pipeline.named_steps else "none",
        "classes": [str(value) for value in forest.classes_],
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    options = "".join(f'<option value="trees/tree-{row["tree"]:04d}.svg">Albero {row["tree"]} - profondita {row["depth"]}, {row["leaves"]} foglie</option>' for row in rows)
    page = f'''<!doctype html><html lang="it"><meta charset="utf-8"><title>Ispezione Random Forest</title>
<style>body{{font:16px system-ui;margin:28px;color:#182230;background:#f5f7fb}}table{{border-collapse:collapse;background:white}}td,th{{padding:7px 14px;border:1px solid #ddd;text-align:left}}iframe{{width:100%;height:650px;border:1px solid #98a2b3;background:white}}select{{font:inherit;padding:8px}}code{{overflow-wrap:anywhere}}</style>
<h1>Random Forest — {escape(run_dir.name)}</h1>
<p>{len(rows)} alberi; {len(names)} feature. Profondita effettiva: min {summary['depth_min']}, media {summary['depth_mean']:.2f}, max {summary['depth_max']}.
Limite max_depth configurato: {forest.max_depth}. Scaler salvato: {escape(summary['scaler'])}.</p>
<p>Classi codificate: {escape(', '.join(summary['classes']))}. Nei nostri tuning 1 indica la classe positiva configurata.</p>
<p>Le soglie originali sono riconvertite usando lo scaler salvato, senza modificare il modello.
Per valori mancanti il confronto riguarda il valore imputato. I campioni nei nodi sono quelli di training/bootstrap, non di test.</p>
<h2>Alberi</h2><p>Radice a profondita 0. Diagrammi limitati a profondita {max_depth}; statistiche e <a href="nodes.csv">nodes.csv</a> includono tutti i nodi.
Scorri il diagramma per leggere i rami; passa sulle feature per il nome completo.</p>
<select aria-label="Scegli albero" onchange="document.getElementById('tree').src=this.value;document.getElementById('svg-link').href=this.value">{options}</select>
<a id="svg-link" href="trees/tree-0001.svg" target="_blank">Apri SVG separato</a>
<iframe id="tree" title="Albero decisionale" src="trees/tree-0001.svg"></iframe>
<details><summary>Statistiche di tutti gli alberi</summary>{stats.to_html(index=False)}</details>
<h2>Importanza delle feature (MDI)</h2><p>Riduzione di impurita sul training: non e' importanza causale o validata su pazienti esterni e puo' essere distorta.
Per la permutation importance su validation fold usa il comando analysis importance.</p>
{importance.head(30).to_html(index=False, float_format=lambda value: f"{value:.6f}")}
<p>Prime 30 feature; elenco completo: <a href="feature_importances.csv">feature_importances.csv</a>.
Statistiche: <a href="trees.csv">trees.csv</a>. Provenienza: <a href="summary.json">summary.json</a>.</p></html>'''
    (report / "index.html").write_text(page, encoding="utf-8")
    return report


def run_forest_inspection(*, run_dir: Path | None, tuning_config_path: Path, output_dir: Path | None = None, max_depth: int = 4) -> Path:
    console = Console()
    console.print("Carica esclusivamente modelli joblib fidati, prodotti dai tuoi tuning.")
    if run_dir is None:
        runs = discover_forest_runs(load_tuning_configuration(tuning_config_path).output_dir)
        try:
            run_dir = choice(
                message="Scegli il tuning della Random Forest (piu' recente per primo):",
                options=[(path, path.name) for path in runs], default=runs[0],
                bottom_toolbar="Frecce: cambia | Invio: conferma | Ctrl+C: annulla",
                show_frame=True, style=SELECTOR_STYLE,
            )
        except (KeyboardInterrupt, EOFError) as error:
            raise RuntimeError("Ispezione annullata") from error
    report = inspect_forest(run_dir, output_dir=output_dir, max_depth=max_depth)
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    table = Table(title="RANDOM FOREST SALVATA", show_header=False)
    for name in ("trees", "features", "configured_max_depth", "depth_min", "depth_mean", "depth_max", "scaler"):
        table.add_row(name, str(summary[name]))
    console.print(table)
    console.print(f"Apri nel browser: {report / 'index.html'}", markup=False)
    return report
