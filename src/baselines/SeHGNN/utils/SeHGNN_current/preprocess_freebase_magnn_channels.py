#!/usr/bin/env python3
"""Preprocess MAGNN-aligned semantic channels for SeHGNN on Freebase NC.

Three channel-set flavors are supported:

* k: independently build native SeHGNN channels up to K on each input variant.
* full_k: build a union graph over the considered dataset variants and generate
  every schema-valid semantic channel up to K hops.
* restricted_k: use the same union graph, but keep only semantic channels that
  occur in the MAGNN union/skip mapping for the considered variants.

Only restricted_k reads MAGNN generated metapath definitions.

The default ``type`` channel identity matches native SeHGNN key semantics: a
channel is identified by its node-type sequence. ``relation`` identity is also
available for diagnostics and preserves relation-level MAGNN distinctions, but
can grow combinatorially for full_k.

The output matrices have shape ``[#BOOK, #source_nodes_of_channel_type]``. They
are consumed as sparse feature selectors by SeHGNN's featureless Freebase path:
``channel_matrix @ trainable_source_type_embedding``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

DEFAULT_SPLIT_SEED = 1566911444
DEFAULT_TARGET_TYPE = 0
DEFAULT_NUM_CLASSES = 8

# This file is intended to live in src/baselines/SeHGNN (the SlotGAT/SeHGNN
# baseline folder in this repository layout).  These defaults match the paths
# requested for running from that folder.
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (BASELINE_DIR / "../../../../../data").resolve()
DEFAULT_VARIANTS_ROOT = DEFAULT_DATA_ROOT / "dataset_variant_3hops_filtered"
DEFAULT_MAGNN_PREPROCESS_ROOT = (
    BASELINE_DIR
    / "../MAGNN/preprocess_scripts/freebase/full_magnn_preprocess_scripts"
).resolve()
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "preprocessed/sehgnn_freebase_magnn"

PIPELINES: Dict[str, Dict[str, Any]] = {
    "up_to_exact_2": {
        "aliases": ("exact_2", "exact2", "up-to-exact-2"),
        "variants": ("unchanged", "exact_2"),
        "up_to_variant": "exact_2",
    },
    "full": {
        "aliases": ("all", "range_2_3", "full_pipeline"),
        # The attached MAGNN bundle contains these four variants. The CLI also
        # accepts --variants for adding another variant without code changes.
        "variants": ("unchanged", "exact_2", "exact_3", "range_2_3"),
        "up_to_variant": "range_2_3",
    },
}


@dataclass(frozen=True)
class RelationSchema:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class DirectedStep:
    relation_name: str
    direction: int
    src_type: int
    dst_type: int

    @property
    def token(self) -> str:
        return f"{self.direction:+d}:{self.relation_name}"


@dataclass
class ChannelRecord:
    model_key: str
    matrix_file: str
    source_type: int
    hop_count: int
    identity_mode: str
    node_type_path: List[int]
    semantic_signatures: List[List[str]]
    dependency_signatures: List[List[str]]
    source_variants: List[str]
    aggregation: str
    nnz: int
    shape: List[int]
    raw_magnn_channel_count: int = 0
    used_dependency_fallbacks: int = 0


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def resolve_pipeline(name: str) -> Tuple[str, Dict[str, Any]]:
    normalized = name.strip().lower().replace("-", "_")
    for canonical, spec in PIPELINES.items():
        choices = {canonical, *(x.replace("-", "_") for x in spec["aliases"])}
        if normalized in choices:
            return canonical, spec
    raise ValueError(f"Unknown pipeline {name!r}; choose one of {sorted(PIPELINES)}")


def parse_info(info_path: Path) -> Tuple[Dict[int, str], Dict[int, RelationSchema], Dict[int, str]]:
    """Parse HGB ``info.dat`` node and relation schema sections."""
    node_type_names: Dict[int, str] = {}
    relation_by_id: Dict[int, RelationSchema] = {}
    relation_name_by_id: Dict[int, str] = {}
    section: Optional[str] = None
    with info_path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped == "node.dat":
                section = "nodes"
                continue
            if stripped == "link.dat":
                section = "links"
                continue
            if stripped in {"label.dat", "label.dat.test"}:
                section = "labels"
                continue
            if (
                stripped.startswith("TYPE\tMEANING")
                or stripped.startswith("LINK\tSTART\tEND\tMEANING")
                or stripped.startswith("TYPE\tCLASS\tMEANING")
                or stripped.startswith("Attribute Dimension:")
                or stripped.startswith("Targeting:")
                or stripped.startswith("---")
            ):
                continue
            parts = [p for p in stripped.split("\t") if p != ""]
            if section == "nodes" and len(parts) >= 2:
                node_type_names[int(parts[0])] = parts[-1]
            elif section == "links" and len(parts) >= 4:
                rid = int(parts[0])
                schema = RelationSchema(parts[-1], int(parts[1]), int(parts[2]))
                relation_by_id[rid] = schema
                relation_name_by_id[rid] = schema.name
    if not relation_by_id:
        raise ValueError(f"No relation schema was parsed from {info_path}")
    return node_type_names, relation_by_id, relation_name_by_id


def read_nodes(node_path: Path) -> np.ndarray:
    rows: List[Tuple[int, int]] = []
    max_id = -1
    with node_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad node.dat line {line_no} in {node_path}")
            nid, ntype = int(parts[0]), int(parts[2])
            rows.append((nid, ntype))
            max_id = max(max_id, nid)
    node_types = np.full(max_id + 1, -1, dtype=np.int64)
    for nid, ntype in rows:
        node_types[nid] = ntype
    if np.any(node_types < 0):
        missing = int((node_types < 0).sum())
        raise ValueError(f"{node_path} has {missing} missing/non-contiguous node ids")
    return node_types


def read_links(link_path: Path, relation_by_id: Mapping[int, RelationSchema]) -> Dict[str, List[Tuple[int, int]]]:
    edges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    with link_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad link.dat line {line_no} in {link_path}")
            src, dst, rid = int(parts[0]), int(parts[1]), int(parts[2])
            if rid not in relation_by_id:
                raise KeyError(f"Relation id {rid} in {link_path}:{line_no} is absent from info.dat")
            edges[relation_by_id[rid].name].append((src, dst))
    return edges


def read_labels(label_path: Path, target_type: int, num_classes: int) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"Bad label.dat line {line_no} in {label_path}")
            nid, ntype, label = int(parts[0]), int(parts[2]), int(parts[3])
            if ntype == target_type and 0 <= label < num_classes:
                rows.append((nid, label))
    if not rows:
        raise ValueError(f"No valid target-type labels found in {label_path}")
    return rows



def validate_ordered_labels_across_variants(
    variants_root: Path,
    variants: Sequence[str],
    target_type: int,
    num_classes: int,
) -> Dict[str, Any]:
    """Require identical ordered label rows so seeded splits match MAGNN exactly.

    The generated MAGNN scripts preserve label.dat order when they call
    train_test_split. Therefore matching only the label set is insufficient:
    all considered variants must expose the same valid (global_id, class) rows
    in the same order.
    """
    reference_variant = "unchanged" if "unchanged" in variants else str(variants[0])
    reference = read_labels(
        variants_root / reference_variant / "label.dat", target_type, num_classes
    )
    per_variant: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        rows = read_labels(
            variants_root / str(variant) / "label.dat", target_type, num_classes
        )
        same_order = rows == reference
        same_set = set(rows) == set(reference)
        per_variant[str(variant)] = {
            "num_valid_labels": len(rows),
            "same_order_as_reference": same_order,
            "same_set_as_reference": same_set,
        }
        if not same_order:
            first_mismatch = next(
                (
                    i
                    for i, (left, right) in enumerate(zip(reference, rows))
                    if left != right
                ),
                min(len(reference), len(rows)),
            )
            reference_item = reference[first_mismatch] if first_mismatch < len(reference) else None
            variant_item = rows[first_mismatch] if first_mismatch < len(rows) else None
            raise RuntimeError(
                f"Ordered label rows differ between {reference_variant} and {variant} "
                f"at position {first_mismatch}: {reference_item} vs {variant_item}. "
                "MAGNN preserves label.dat order during seeded splitting, so the "
                "splits cannot be guaranteed identical until the label files match."
            )
    return {
        "reference_variant": reference_variant,
        "num_valid_labels": len(reference),
        "ordered_labels_identical": True,
        "per_variant": per_variant,
    }

def make_local_index(node_types: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    by_type: Dict[int, np.ndarray] = {}
    global_to_local = np.full(len(node_types), -1, dtype=np.int64)
    for ntype in sorted(set(int(x) for x in node_types.tolist())):
        ids = np.where(node_types == ntype)[0].astype(np.int64)
        by_type[ntype] = ids
        global_to_local[ids] = np.arange(len(ids), dtype=np.int64)
    return by_type, global_to_local


def stratified_split(
    labeled_local_ids: Sequence[int], labels: Sequence[int], seed: int,
    train_ratio: float = 0.6, val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the RGCN two-stage stratified 60/20/20 split."""
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    ids = np.asarray(labeled_local_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    train_idx, rest_idx, _, rest_y = train_test_split(
        ids,
        y,
        test_size=val_ratio + test_ratio,
        stratify=y,
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=relative_test,
        stratify=rest_y,
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(sums, dtype=np.float32)
    nz = sums != 0
    inv[nz] = 1.0 / sums[nz]
    return (sp.diags(inv, format="csr") @ matrix).tocsr()


def binarize(matrix: sp.spmatrix) -> sp.csr_matrix:
    out = matrix.tocsr().astype(np.float32)
    out.sum_duplicates()
    out.eliminate_zeros()
    if out.nnz:
        out.data[:] = 1.0
    return out


def boolean_product(left: sp.csr_matrix, right: sp.csr_matrix) -> sp.csr_matrix:
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"Sparse product shape mismatch: {left.shape} @ {right.shape}")
    return binarize(left @ right)


def normalized_product(left: sp.csr_matrix, right: sp.csr_matrix) -> sp.csr_matrix:
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"Sparse product shape mismatch: {left.shape} @ {right.shape}")
    out = (left @ right).tocsr().astype(np.float32)
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def locate_magnn_root(root: Path) -> Path:
    candidates = [root, root / "full_magnn_preprocess_scripts"]
    for candidate in candidates:
        if (candidate / "unchanged").is_dir():
            return candidate
    matches = list(root.rglob("unchanged/exact_2/union/center/preprocess_freebase_node.py"))
    if len(matches) == 1:
        return matches[0].parents[4]
    raise FileNotFoundError(
        f"Could not locate the MAGNN generated-script root beneath {root}; expected an unchanged/ folder"
    )


def extract_python_literals(path: Path, names: Iterable[str]) -> Dict[str, Any]:
    wanted = set(names)
    found: Dict[str, Any] = {}
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for stmt in module.body:
        target: Optional[ast.Name] = None
        value: Optional[ast.AST] = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target, value = stmt.target, stmt.value
        if target is not None and value is not None and target.id in wanted:
            found[target.id] = ast.literal_eval(value)
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"Missing constants {sorted(missing)} in {path}")
    return found


def parse_signature_token(token: str) -> Tuple[int, str]:
    match = re.fullmatch(r"([+-]1):(.+)", str(token))
    if not match:
        raise ValueError(f"Bad MAGNN signature token: {token!r}")
    return int(match.group(1)), match.group(2)


def make_model_key(index: int, source_type: int, identity: bool = False) -> str:
    if not (0 <= int(source_type) <= 9):
        raise ValueError(
            "The repository SeHGNN Freebase branch selects source embeddings using key[-1]; "
            f"source type {source_type} is not a single decimal digit"
        )
    return str(source_type) if identity else f"C{index:06d}_T{source_type}"


class UnionGraph:
    def __init__(
        self,
        node_types: np.ndarray,
        relation_schemas: Mapping[str, RelationSchema],
        relation_edges: Mapping[str, Sequence[Tuple[int, int]]],
    ) -> None:
        self.node_types = node_types
        self.relation_schemas = dict(relation_schemas)
        self.type_global_ids, self.global_to_local = make_local_index(node_types)
        self.forward_raw: Dict[str, sp.csr_matrix] = {}
        for name, schema in sorted(self.relation_schemas.items()):
            n_src = len(self.type_global_ids[schema.start])
            n_dst = len(self.type_global_ids[schema.end])
            rows: List[int] = []
            cols: List[int] = []
            for src, dst in relation_edges.get(name, []):
                if not (0 <= src < len(node_types) and 0 <= dst < len(node_types)):
                    raise ValueError(f"Out-of-range edge ({src}, {dst}) for relation {name}")
                if int(node_types[src]) != schema.start or int(node_types[dst]) != schema.end:
                    raise ValueError(
                        f"Edge ({src}, {dst}) for {name} has node types "
                        f"({node_types[src]}, {node_types[dst]}) but schema is ({schema.start}, {schema.end})"
                    )
                rows.append(int(self.global_to_local[src]))
                cols.append(int(self.global_to_local[dst]))
            values = np.ones(len(rows), dtype=np.float32)
            raw = sp.coo_matrix((values, (rows, cols)), shape=(n_src, n_dst), dtype=np.float32).tocsr()
            self.forward_raw[name] = binarize(raw)

        self.directed_steps_by_src: Dict[int, List[DirectedStep]] = defaultdict(list)
        for name, schema in sorted(self.relation_schemas.items()):
            self.directed_steps_by_src[schema.start].append(
                DirectedStep(name, +1, schema.start, schema.end)
            )
            self.directed_steps_by_src[schema.end].append(
                DirectedStep(name, -1, schema.end, schema.start)
            )
        for src in self.directed_steps_by_src:
            self.directed_steps_by_src[src].sort(key=lambda s: (s.dst_type, s.relation_name, s.direction))

        self._type_step_raw: Dict[Tuple[int, int], sp.csr_matrix] = {}
        grouped: Dict[Tuple[int, int], List[sp.csr_matrix]] = defaultdict(list)
        for name, schema in self.relation_schemas.items():
            raw = self.forward_raw[name]
            grouped[(schema.start, schema.end)].append(raw)
            grouped[(schema.end, schema.start)].append(raw.T.tocsr())
        for pair, matrices in grouped.items():
            total = matrices[0].copy()
            for matrix in matrices[1:]:
                total = total + matrix
            self._type_step_raw[pair] = binarize(total)
        self.type_steps_by_src: Dict[int, List[int]] = defaultdict(list)
        for src, dst in sorted(self._type_step_raw):
            self.type_steps_by_src[src].append(dst)

    def directed_raw(self, relation_name: str, direction: int) -> Tuple[sp.csr_matrix, int, int]:
        if relation_name not in self.relation_schemas:
            raise KeyError(relation_name)
        schema = self.relation_schemas[relation_name]
        if direction == +1:
            return self.forward_raw[relation_name], schema.start, schema.end
        if direction == -1:
            return self.forward_raw[relation_name].T.tocsr(), schema.end, schema.start
        raise ValueError(f"direction must be +1 or -1, got {direction}")

    def directed_normalized(self, relation_name: str, direction: int) -> Tuple[sp.csr_matrix, int, int]:
        raw, src, dst = self.directed_raw(relation_name, direction)
        return row_normalize(raw), src, dst

    def type_raw(self, src_type: int, dst_type: int) -> sp.csr_matrix:
        return self._type_step_raw[(src_type, dst_type)]

    def type_normalized(self, src_type: int, dst_type: int) -> sp.csr_matrix:
        return row_normalize(self.type_raw(src_type, dst_type))


def build_union_graph(
    variants_root: Path,
    variants: Sequence[str],
) -> Tuple[UnionGraph, Dict[int, str], Dict[str, Any]]:
    base_node_types: Optional[np.ndarray] = None
    merged_schemas: Dict[str, RelationSchema] = {}
    merged_edges: Dict[str, set[Tuple[int, int]]] = defaultdict(set)
    node_type_names: Dict[int, str] = {}
    per_variant: Dict[str, Any] = {}

    for variant in variants:
        variant_dir = variants_root / variant
        for required in ("node.dat", "link.dat", "info.dat"):
            if not (variant_dir / required).exists():
                raise FileNotFoundError(f"Missing {variant_dir / required}")
        names, relation_by_id, _ = parse_info(variant_dir / "info.dat")
        if names:
            node_type_names.update(names)
        node_types = read_nodes(variant_dir / "node.dat")
        if base_node_types is None:
            base_node_types = node_types
        elif not np.array_equal(base_node_types, node_types):
            raise ValueError(
                f"Node ids/types differ in variant {variant}; union preprocessing requires aligned nodes"
            )
        edges = read_links(variant_dir / "link.dat", relation_by_id)
        for schema in relation_by_id.values():
            old = merged_schemas.get(schema.name)
            if old is not None and old != schema:
                raise ValueError(f"Conflicting schemas for relation {schema.name}: {old} vs {schema}")
            merged_schemas[schema.name] = schema
        for name, pairs in edges.items():
            merged_edges[name].update(pairs)
        per_variant[variant] = {
            "num_relations": len(relation_by_id),
            "num_edges": sum(len(x) for x in edges.values()),
        }

    assert base_node_types is not None
    graph = UnionGraph(base_node_types, merged_schemas, merged_edges)
    audit = {
        "variants": per_variant,
        "union_num_relations": len(merged_schemas),
        "union_num_edges_by_relation_deduplicated": {
            name: len(pairs) for name, pairs in sorted(merged_edges.items())
        },
        "union_num_edges_total_deduplicated": int(sum(len(x) for x in merged_edges.values())),
    }
    return graph, node_type_names, audit


def audit_union_against_endpoint(
    graph: UnionGraph,
    variants_root: Path,
    endpoint_variant: str,
) -> Dict[str, Any]:
    """Compare the explicit union graph with one endpoint input graph exactly.

    Equality requires identical relation-name/schema sets and identical
    deduplicated directed edges for every relation.
    """
    endpoint_dir = variants_root / endpoint_variant
    names, relation_by_id, _ = parse_info(endpoint_dir / "info.dat")
    del names
    endpoint_node_types = read_nodes(endpoint_dir / "node.dat")
    if not np.array_equal(graph.node_types, endpoint_node_types):
        return {
            "endpoint_variant": endpoint_variant,
            "union_equivalent_to_endpoint_variant": False,
            "node_types_equal": False,
            "reason": "node ids/types differ",
        }
    endpoint_edges = read_links(endpoint_dir / "link.dat", relation_by_id)
    endpoint_schemas = {schema.name: schema for schema in relation_by_id.values()}
    union_names = set(graph.relation_schemas)
    endpoint_names = set(endpoint_schemas)
    all_names = sorted(union_names | endpoint_names)
    per_relation: Dict[str, Any] = {}
    equivalent = union_names == endpoint_names
    schema_mismatches: List[str] = []
    for name in all_names:
        union_schema = graph.relation_schemas.get(name)
        endpoint_schema = endpoint_schemas.get(name)
        if union_schema != endpoint_schema:
            equivalent = False
            schema_mismatches.append(name)
        schema = union_schema or endpoint_schema
        assert schema is not None
        shape = (
            len(graph.type_global_ids[schema.start]),
            len(graph.type_global_ids[schema.end]),
        )
        union_matrix = graph.forward_raw.get(name, sp.csr_matrix(shape, dtype=np.float32))
        rows: List[int] = []
        cols: List[int] = []
        for src, dst in endpoint_edges.get(name, []):
            rows.append(int(graph.global_to_local[src]))
            cols.append(int(graph.global_to_local[dst]))
        endpoint_matrix = binarize(
            sp.coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=shape,
                dtype=np.float32,
            ).tocsr()
        )
        union_only = binarize(union_matrix - union_matrix.multiply(endpoint_matrix))
        endpoint_only = binarize(endpoint_matrix - endpoint_matrix.multiply(union_matrix))
        same = union_only.nnz == 0 and endpoint_only.nnz == 0
        equivalent = equivalent and same
        per_relation[name] = {
            "union_edges": int(union_matrix.nnz),
            "endpoint_edges": int(endpoint_matrix.nnz),
            "union_only_edges": int(union_only.nnz),
            "endpoint_only_edges": int(endpoint_only.nnz),
            "equal": bool(same and union_schema == endpoint_schema),
        }
    return {
        "endpoint_variant": endpoint_variant,
        "node_types_equal": True,
        "union_relation_names": sorted(union_names),
        "endpoint_relation_names": sorted(endpoint_names),
        "union_only_relations": sorted(union_names - endpoint_names),
        "endpoint_only_relations": sorted(endpoint_names - union_names),
        "schema_mismatches": schema_mismatches,
        "per_relation": per_relation,
        "union_equivalent_to_endpoint_variant": bool(equivalent),
    }


def type_path_from_signature(
    graph: UnionGraph,
    signature: Sequence[str],
    target_type: int,
) -> List[int]:
    current_type = int(target_type)
    path = [current_type]
    for token in signature:
        direction, relation_name = parse_signature_token(token)
        if relation_name not in graph.relation_schemas:
            raise KeyError(relation_name)
        schema = graph.relation_schemas[relation_name]
        if direction == +1:
            src_type, dst_type = schema.start, schema.end
        else:
            src_type, dst_type = schema.end, schema.start
        if current_type != src_type:
            raise ValueError(
                f"Signature {signature} is not continuous at {token}: current type {current_type}, "
                f"step starts at {src_type}"
            )
        current_type = dst_type
        path.append(current_type)
    return path


def matrix_from_signature(
    graph: UnionGraph,
    signature: Sequence[str],
    target_type: int,
    *,
    binary_reachability: bool,
) -> Tuple[sp.csr_matrix, List[int]]:
    current_type = int(target_type)
    path = [current_type]
    result = sp.identity(
        len(graph.type_global_ids[current_type]), format="csr", dtype=np.float32
    )
    for token in signature:
        direction, relation_name = parse_signature_token(token)
        raw, src_type, dst_type = graph.directed_raw(relation_name, direction)
        if current_type != src_type:
            raise ValueError(
                f"Signature {signature} is not continuous at {token}: current type {current_type}, "
                f"step starts at {src_type}"
            )
        if binary_reachability:
            result = boolean_product(result, raw)
        else:
            result = normalized_product(result, row_normalize(raw))
        current_type = dst_type
        path.append(current_type)
    return result, path


def enumerate_full_type_channels(
    graph: UnionGraph,
    target_type: int,
    k: int,
    max_channels: int,
) -> List[Tuple[Tuple[int, ...], sp.csr_matrix]]:
    target_count = len(graph.type_global_ids[target_type])
    identity = sp.identity(target_count, format="csr", dtype=np.float32)
    out: List[Tuple[Tuple[int, ...], sp.csr_matrix]] = [((target_type,), identity)]
    frontier: List[Tuple[Tuple[int, ...], sp.csr_matrix]] = [((target_type,), identity)]
    for _hop in range(1, k + 1):
        next_frontier: List[Tuple[Tuple[int, ...], sp.csr_matrix]] = []
        for path, prefix in frontier:
            src_type = path[-1]
            for dst_type in graph.type_steps_by_src.get(src_type, []):
                matrix = normalized_product(prefix, graph.type_normalized(src_type, dst_type))
                next_frontier.append((path + (dst_type,), matrix))
                if len(out) + len(next_frontier) > max_channels:
                    raise RuntimeError(
                        f"full_k would exceed --max-channels={max_channels} at hop {_hop}; "
                        "lower K or explicitly raise the cap"
                    )
        out.extend(next_frontier)
        frontier = next_frontier
    return out


def enumerate_full_relation_channels(
    graph: UnionGraph,
    target_type: int,
    k: int,
    max_channels: int,
) -> List[Tuple[Tuple[str, ...], Tuple[int, ...], sp.csr_matrix]]:
    target_count = len(graph.type_global_ids[target_type])
    identity = sp.identity(target_count, format="csr", dtype=np.float32)
    out: List[Tuple[Tuple[str, ...], Tuple[int, ...], sp.csr_matrix]] = [
        (tuple(), (target_type,), identity)
    ]
    frontier = [(tuple(), (target_type,), identity)]
    for hop in range(1, k + 1):
        next_frontier = []
        for signature, type_path, prefix in frontier:
            src_type = type_path[-1]
            for step in graph.directed_steps_by_src.get(src_type, []):
                normalized, _, _ = graph.directed_normalized(step.relation_name, step.direction)
                matrix = normalized_product(prefix, normalized)
                next_frontier.append(
                    (signature + (step.token,), type_path + (step.dst_type,), matrix)
                )
                if len(out) + len(next_frontier) > max_channels:
                    raise RuntimeError(
                        f"relation-aware full_k would exceed --max-channels={max_channels} at hop {hop}; "
                        "use type identity, lower K, or explicitly raise the cap"
                    )
        out.extend(next_frontier)
        frontier = next_frontier
    return out


def load_magnn_union_metapaths(
    magnn_root: Path,
    up_to_variant: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, str], Path]:
    root = locate_magnn_root(magnn_root)
    script = root / "unchanged" / up_to_variant / "union" / "center" / "preprocess_freebase_node.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Restricted-K requires the MAGNN union/center script, but it was not found: {script}"
        )
    literals = extract_python_literals(script, {"metapath_defs", "node_type_names"})
    return list(literals["metapath_defs"]), dict(literals["node_type_names"]), script


def choose_realizable_signature(
    graph: UnionGraph,
    spec: Mapping[str, Any],
    target_type: int,
) -> Tuple[List[str], bool, List[int]]:
    candidates = [
        (list(spec.get("semantic_signature") or []), False),
        (list(spec.get("dependency_signature") or []), True),
    ]
    errors: List[str] = []
    for signature, fallback in candidates:
        if not signature:
            continue
        if any(parse_signature_token(tok)[1] not in graph.relation_schemas for tok in signature):
            missing = [
                parse_signature_token(tok)[1]
                for tok in signature
                if parse_signature_token(tok)[1] not in graph.relation_schemas
            ]
            errors.append(f"missing relations {missing}")
            continue
        try:
            path = type_path_from_signature(graph, signature, target_type)
            return signature, fallback, path
        except Exception as exc:  # keep the more informative combined error below
            errors.append(str(exc))
    raise ValueError(
        f"Neither semantic nor dependency signature is realizable for MAGNN metapath "
        f"{spec.get('name', '<unnamed>')}: {'; '.join(errors)}"
    )


def restricted_channel_groups(
    graph: UnionGraph,
    metapath_defs: Sequence[Mapping[str, Any]],
    target_type: int,
    k: int,
    identity_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: MutableMapping[Any, List[Dict[str, Any]]] = defaultdict(list)
    skipped_over_k = 0
    skipped_wrong_center = 0
    for spec in metapath_defs:
        semantic_path = tuple(int(x) for x in spec.get("semantic_node_metapath", spec.get("node_metapath", [])))
        if not semantic_path or semantic_path[0] != target_type or semantic_path[-1] != target_type:
            skipped_wrong_center += 1
            continue
        hop_count = len(semantic_path) - 1
        if hop_count > k:
            skipped_over_k += 1
            continue
        signature, used_fallback, realized_path = choose_realizable_signature(graph, spec, target_type)
        if tuple(realized_path) != semantic_path and not used_fallback:
            raise ValueError(
                f"MAGNN semantic signature realizes node path {realized_path}, expected {semantic_path}"
            )
        if identity_mode == "type":
            group_key: Any = semantic_path
        elif identity_mode == "relation":
            group_key = tuple(signature)
        else:
            raise ValueError(identity_mode)
        groups[group_key].append(
            {
                "spec": dict(spec),
                "signature": signature,
                "used_fallback": used_fallback,
                "semantic_path": semantic_path,
            }
        )

    prepared: List[Dict[str, Any]] = []
    for group_key in sorted(groups, key=lambda x: repr(x)):
        members = groups[group_key]
        combined: Optional[sp.csr_matrix] = None
        all_semantic: List[List[str]] = []
        all_dependency: List[List[str]] = []
        all_variants: set[str] = set()
        fallback_count = 0
        path = list(members[0]["semantic_path"])
        for member in members:
            matrix, _ = matrix_from_signature(
                graph,
                member["signature"],
                target_type,
                binary_reachability=True,
            )
            combined = matrix if combined is None else binarize(combined + matrix)
            spec = member["spec"]
            all_semantic.append(list(spec.get("semantic_signature") or member["signature"]))
            all_dependency.append(list(spec.get("dependency_signature") or []))
            all_variants.update(str(x) for x in spec.get("source_variants", []))
            fallback_count += int(member["used_fallback"])
        assert combined is not None
        prepared.append(
            {
                "group_key": group_key,
                "matrix": row_normalize(combined),
                "source_type": int(path[-1]),
                "node_type_path": path,
                "semantic_signatures": all_semantic,
                "dependency_signatures": all_dependency,
                "source_variants": sorted(all_variants),
                "raw_count": len(members),
                "fallback_count": fallback_count,
            }
        )

    audit = {
        "raw_magnn_metapaths": len(metapath_defs),
        "kept_raw_magnn_metapaths": int(sum(x["raw_count"] for x in prepared)),
        "output_semantic_channels_excluding_identity": len(prepared),
        "skipped_over_k": skipped_over_k,
        "skipped_wrong_center": skipped_wrong_center,
        "collision_groups": int(sum(x["raw_count"] > 1 for x in prepared)),
        "largest_collision_group": int(max([x["raw_count"] for x in prepared] or [0])),
    }
    return prepared, audit


def save_dataset_files(
    output_dir: Path,
    graph: UnionGraph,
    variants_root: Path,
    target_type: int,
    num_classes: int,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Any]:
    target_global = graph.type_global_ids[target_type]
    target_to_local = {int(nid): i for i, nid in enumerate(target_global.tolist())}
    label_rows = read_labels(
        variants_root / "unchanged" / "label.dat", target_type, num_classes
    )
    labels = np.full(len(target_global), -1, dtype=np.int64)
    labeled_local: List[int] = []
    labeled_y: List[int] = []
    for nid, label in label_rows:
        if nid not in target_to_local:
            continue
        local = target_to_local[nid]
        labels[local] = label
        labeled_local.append(local)
        labeled_y.append(label)
    train_idx, val_idx, test_idx = stratified_split(
        labeled_local,
        labeled_y,
        split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    np.savez_compressed(
        output_dir / "dataset.npz",
        node_types=graph.node_types,
        target_global_ids=target_global,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_global_idx=target_global[train_idx],
        val_global_idx=target_global[val_idx],
        test_global_idx=target_global[test_idx],
    )
    np.savez_compressed(
        output_dir / "node_type_global_ids.npz",
        **{f"type_{t}": ids for t, ids in sorted(graph.type_global_ids.items())},
    )
    return {
        "num_target_nodes": len(target_global),
        "num_labeled_target_nodes": len(labeled_local),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "label_counts": {
            str(c): int(np.sum(np.asarray(labeled_y) == c)) for c in range(num_classes)
        },
    }


def read_saved_global_split(output_dir: Path) -> Dict[str, np.ndarray]:
    """Read the global node-id split saved for one preprocessed dataset."""
    with np.load(output_dir / "dataset.npz", allow_pickle=False) as data:
        return {
            "train": np.asarray(data["train_global_idx"], dtype=np.int64),
            "val": np.asarray(data["val_global_idx"], dtype=np.int64),
            "test": np.asarray(data["test_global_idx"], dtype=np.int64),
        }


def assert_global_split_alignment(
    reference: Optional[Mapping[str, np.ndarray]],
    candidate: Mapping[str, np.ndarray],
    *,
    reference_name: str,
    candidate_name: str,
) -> Dict[str, np.ndarray]:
    """Fail fast if two variants/flavors do not use identical global splits."""
    normalized = {
        key: np.asarray(candidate[key], dtype=np.int64)
        for key in ("train", "val", "test")
    }
    if reference is None:
        return normalized
    for split_name in ("train", "val", "test"):
        left = np.asarray(reference[split_name], dtype=np.int64)
        right = normalized[split_name]
        if not np.array_equal(left, right):
            left_only = np.setdiff1d(left, right)
            right_only = np.setdiff1d(right, left)
            raise RuntimeError(
                f"Split mismatch for {split_name}: {reference_name} vs "
                f"{candidate_name}. Sizes are {len(left)} vs {len(right)}; "
                f"reference-only sample={left_only[:10].tolist()}, "
                f"candidate-only sample={right_only[:10].tolist()}."
            )
    return {key: np.asarray(reference[key], dtype=np.int64) for key in reference}


def write_channels(
    output_dir: Path,
    channels: Sequence[Dict[str, Any]],
    *,
    pipeline_name: str,
    variants: Sequence[str],
    up_to_variant: str,
    flavor: str,
    k: int,
    identity_mode: str,
    target_type: int,
    num_classes: int,
    split_seed: int,
    node_type_names: Mapping[int, str],
    graph: UnionGraph,
    graph_audit: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    restricted_audit: Optional[Mapping[str, Any]],
    magnn_source_script: Optional[Path],
) -> None:
    channel_dir = output_dir / "channels"
    if channel_dir.exists():
        shutil.rmtree(channel_dir)
    channel_dir.mkdir(parents=True, exist_ok=True)

    records: List[ChannelRecord] = []
    for index, channel in enumerate(channels):
        source_type = int(channel["source_type"])
        identity = index == 0
        model_key = make_model_key(index, source_type, identity=identity)
        filename = f"{index:06d}_{model_key}.npz"
        matrix = channel["matrix"].tocsr().astype(np.float32)
        sp.save_npz(channel_dir / filename, matrix)
        records.append(
            ChannelRecord(
                model_key=model_key,
                matrix_file=f"channels/{filename}",
                source_type=source_type,
                hop_count=int(channel["hop_count"]),
                identity_mode=identity_mode,
                node_type_path=[int(x) for x in channel["node_type_path"]],
                semantic_signatures=[list(x) for x in channel.get("semantic_signatures", [])],
                dependency_signatures=[list(x) for x in channel.get("dependency_signatures", [])],
                source_variants=[str(x) for x in channel.get("source_variants", [])],
                aggregation=str(channel["aggregation"]),
                nnz=int(matrix.nnz),
                shape=[int(matrix.shape[0]), int(matrix.shape[1])],
                raw_magnn_channel_count=int(channel.get("raw_count", 0)),
                used_dependency_fallbacks=int(channel.get("fallback_count", 0)),
            )
        )

    manifest = {
        "format_version": 1,
        "pipeline": pipeline_name,
        "variant": (str(variants[0]) if flavor == "k" and len(variants) == 1 else None),
        "variants": list(variants),
        "up_to_variant": up_to_variant,
        "flavor": flavor,
        "k": k,
        "channel_identity": identity_mode,
        "target_type": target_type,
        "target_type_name": node_type_names.get(target_type, str(target_type)),
        "num_classes": num_classes,
        "split_seed": split_seed,
        "node_type_names": {str(k): v for k, v in sorted(node_type_names.items())},
        "node_counts": {
            str(t): int(len(ids)) for t, ids in sorted(graph.type_global_ids.items())
        },
        "num_channels": len(records),
        "identity_channel_key": str(target_type),
        "magnn_source_script": str(magnn_source_script) if magnn_source_script else None,
        "channels": [asdict(x) for x in records],
        "graph_audit": _jsonable(graph_audit),
        "dataset_audit": _jsonable(dataset_audit),
        "restricted_audit": _jsonable(restricted_audit) if restricted_audit else None,
    }
    (output_dir / "channels_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )
    summary = {
        "pipeline": pipeline_name,
        "flavor": flavor,
        "k": k,
        "channel_identity": identity_mode,
        "num_channels": len(records),
        "channels_by_hop": {
            str(h): int(sum(r.hop_count == h for r in records))
            for h in sorted(set(r.hop_count for r in records))
        },
        "total_channel_nnz": int(sum(r.nnz for r in records)),
        "zero_nnz_channels": int(sum(r.nnz == 0 for r in records)),
        "dataset": _jsonable(dataset_audit),
        "restricted": _jsonable(restricted_audit) if restricted_audit else None,
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def prepare_full_channels(
    graph: UnionGraph,
    target_type: int,
    k: int,
    identity_mode: str,
    max_channels: int,
) -> List[Dict[str, Any]]:
    channels: List[Dict[str, Any]] = []
    if identity_mode == "type":
        generated = enumerate_full_type_channels(graph, target_type, k, max_channels)
        for path, matrix in generated:
            channels.append(
                {
                    "matrix": matrix,
                    "source_type": path[-1],
                    "hop_count": len(path) - 1,
                    "node_type_path": list(path),
                    "semantic_signatures": [],
                    "dependency_signatures": [],
                    "source_variants": [],
                    "aggregation": "nested_mean_over_union_type_transition_edges",
                }
            )
    else:
        generated = enumerate_full_relation_channels(graph, target_type, k, max_channels)
        for signature, path, matrix in generated:
            channels.append(
                {
                    "matrix": matrix,
                    "source_type": path[-1],
                    "hop_count": len(signature),
                    "node_type_path": list(path),
                    "semantic_signatures": [list(signature)] if signature else [],
                    "dependency_signatures": [],
                    "source_variants": [],
                    "aggregation": "nested_mean_per_relation_sequence",
                }
            )
    return channels


def prepare_restricted_channels(
    graph: UnionGraph,
    target_type: int,
    k: int,
    identity_mode: str,
    metapath_defs: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_count = len(graph.type_global_ids[target_type])
    channels: List[Dict[str, Any]] = [
        {
            "matrix": sp.identity(target_count, format="csr", dtype=np.float32),
            "source_type": target_type,
            "hop_count": 0,
            "node_type_path": [target_type],
            "semantic_signatures": [],
            "dependency_signatures": [],
            "source_variants": [],
            "aggregation": "identity",
        }
    ]
    groups, audit = restricted_channel_groups(
        graph, metapath_defs, target_type, k, identity_mode
    )
    for group in groups:
        channels.append(
            {
                "matrix": group["matrix"],
                "source_type": group["source_type"],
                "hop_count": len(group["node_type_path"]) - 1,
                "node_type_path": group["node_type_path"],
                "semantic_signatures": group["semantic_signatures"],
                "dependency_signatures": group["dependency_signatures"],
                "source_variants": group["source_variants"],
                "aggregation": (
                    "binary_union_of_magnn_semantic_endpoint_pairs_then_row_mean"
                    if identity_mode == "type"
                    else "magnn_semantic_endpoint_pairs_then_row_mean"
                ),
                "raw_count": group["raw_count"],
                "fallback_count": group["fallback_count"],
            }
        )
    return channels, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-variant K, universal full-K, and MAGNN-restricted-K "
            "SeHGNN channels for Freebase NC"
        )
    )
    parser.add_argument(
        "--variants-root",
        type=Path,
        default=DEFAULT_VARIANTS_ROOT,
        help=f"Variant data root (default: {DEFAULT_VARIANTS_ROOT})",
    )
    parser.add_argument(
        "--magnn-preprocess-root",
        type=Path,
        default=DEFAULT_MAGNN_PREPROCESS_ROOT,
        help=(
            "Generated MAGNN preprocess-script root. Only read for restricted_k "
            f"(default: {DEFAULT_MAGNN_PREPROCESS_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Preprocessing output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--pipeline", default="up_to_exact_2")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Override the named pipeline's ordered considered variants",
    )
    parser.add_argument(
        "--up-to-variant",
        default=None,
        help="MAGNN folder used for restricted-K; defaults to the pipeline endpoint",
    )
    parser.add_argument(
        "--flavor",
        choices=["k", "full_k", "restricted_k", "both", "all"],
        default="all",
        help=(
            "k=per-variant native channels; both=full_k+restricted_k; "
            "all=k+full_k+restricted_k"
        ),
    )
    # Generic SeHGNN GitHub default. Dataset-specific exact_2 scripts pass K=2.
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--channel-identity",
        choices=["type", "relation"],
        default="type",
        help="type matches native SeHGNN; relation preserves exact MAGNN relation signatures",
    )
    parser.add_argument("--target-type", type=int, default=DEFAULT_TARGET_TYPE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-channels", type=int, default=10_000)
    parser.add_argument(
        "--require-endpoint-equals-union",
        action="store_true",
        help=(
            "Fail unless the explicit union graph exactly equals the up-to/endpoint "
            "variant. Useful for validating exact_2 as the universal graph."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.k < 0:
        raise ValueError("--k must be nonnegative")
    pipeline_name, pipeline_spec = resolve_pipeline(args.pipeline)
    variants = tuple(args.variants or pipeline_spec["variants"])
    up_to_variant = args.up_to_variant or pipeline_spec["up_to_variant"]
    if "unchanged" not in variants:
        raise ValueError("The considered variants must include unchanged")
    if up_to_variant not in variants:
        raise ValueError(f"up-to variant {up_to_variant!r} is not in {variants}")

    label_alignment_audit = validate_ordered_labels_across_variants(
        args.variants_root, variants, args.target_type, args.num_classes
    )

    if args.flavor == "all":
        flavors = ["k", "full_k", "restricted_k"]
    elif args.flavor == "both":
        flavors = ["full_k", "restricted_k"]
    else:
        flavors = [args.flavor]

    base_output = (
        args.output_root
        / pipeline_name
        / f"k{args.k}"
        / f"{args.channel_identity}_channels"
    )

    # Every generated K/fullK/restrictedK dataset must use exactly the same
    # global BOOK-node split. This catches node-ordering or label-filtering
    # differences even when the same random seed is supplied.
    reference_global_split: Optional[Dict[str, np.ndarray]] = None
    reference_split_name = ""

    # K: build channels independently on each original/transformed graph.  No
    # MAGNN definitions are read for this flavor.
    if "k" in flavors:
        for variant in variants:
            graph, node_type_names, graph_audit = build_union_graph(
                args.variants_root, [variant]
            )
            graph_audit = dict(graph_audit)
            graph_audit["label_alignment"] = label_alignment_audit
            output_dir = base_output / "k" / variant
            _prepare_output_dir(output_dir, args.overwrite)
            dataset_audit = save_dataset_files(
                output_dir,
                graph,
                args.variants_root,
                args.target_type,
                args.num_classes,
                args.split_seed,
                args.train_ratio,
                args.val_ratio,
            )
            candidate_split = read_saved_global_split(output_dir)
            candidate_name = f"k/{variant}"
            if reference_global_split is None:
                reference_global_split = assert_global_split_alignment(
                    None, candidate_split,
                    reference_name=candidate_name, candidate_name=candidate_name,
                )
                reference_split_name = candidate_name
            else:
                assert_global_split_alignment(
                    reference_global_split, candidate_split,
                    reference_name=reference_split_name, candidate_name=candidate_name,
                )
            channels = prepare_full_channels(
                graph,
                args.target_type,
                args.k,
                args.channel_identity,
                args.max_channels,
            )
            write_channels(
                output_dir,
                channels,
                pipeline_name=pipeline_name,
                variants=[variant],
                up_to_variant=variant,
                flavor="k",
                k=args.k,
                identity_mode=args.channel_identity,
                target_type=args.target_type,
                num_classes=args.num_classes,
                split_seed=args.split_seed,
                node_type_names=node_type_names,
                graph=graph,
                graph_audit=graph_audit,
                dataset_audit=dataset_audit,
                restricted_audit=None,
                magnn_source_script=None,
            )
            print(f"Saved k/{variant}: {output_dir} ({len(channels)} channels)")

    # fullK/restrictedK share one explicit union graph.
    universal_flavors = [f for f in flavors if f in {"full_k", "restricted_k"}]
    if universal_flavors:
        graph, parsed_node_type_names, graph_audit = build_union_graph(
            args.variants_root, variants
        )
        endpoint_audit = audit_union_against_endpoint(
            graph, args.variants_root, up_to_variant
        )
        graph_audit = dict(graph_audit)
        graph_audit.update(endpoint_audit)
        graph_audit["label_alignment"] = label_alignment_audit
        if (
            args.require_endpoint_equals_union
            and not endpoint_audit["union_equivalent_to_endpoint_variant"]
        ):
            raise RuntimeError(
                f"The union of {variants} does not exactly equal endpoint "
                f"{up_to_variant}. See graph audit details above/in a run without "
                "--require-endpoint-equals-union."
            )

        metapath_defs: List[Dict[str, Any]] = []
        magnn_node_type_names: Dict[int, str] = {}
        magnn_script: Optional[Path] = None
        if "restricted_k" in universal_flavors:
            # This is intentionally the only point at which MAGNN files are read.
            metapath_defs, magnn_node_type_names, magnn_script = (
                load_magnn_union_metapaths(
                    args.magnn_preprocess_root, up_to_variant
                )
            )

        node_type_names = dict(parsed_node_type_names)
        node_type_names.update(
            {int(k): str(v) for k, v in magnn_node_type_names.items()}
        )

        for flavor in universal_flavors:
            output_dir = base_output / flavor
            _prepare_output_dir(output_dir, args.overwrite)
            dataset_audit = save_dataset_files(
                output_dir,
                graph,
                args.variants_root,
                args.target_type,
                args.num_classes,
                args.split_seed,
                args.train_ratio,
                args.val_ratio,
            )
            candidate_split = read_saved_global_split(output_dir)
            candidate_name = flavor
            if reference_global_split is None:
                reference_global_split = assert_global_split_alignment(
                    None, candidate_split,
                    reference_name=candidate_name, candidate_name=candidate_name,
                )
                reference_split_name = candidate_name
            else:
                assert_global_split_alignment(
                    reference_global_split, candidate_split,
                    reference_name=reference_split_name, candidate_name=candidate_name,
                )
            restricted_audit: Optional[Dict[str, Any]] = None
            source_script: Optional[Path] = None
            if flavor == "full_k":
                channels = prepare_full_channels(
                    graph,
                    args.target_type,
                    args.k,
                    args.channel_identity,
                    args.max_channels,
                )
            else:
                channels, restricted_audit = prepare_restricted_channels(
                    graph,
                    args.target_type,
                    args.k,
                    args.channel_identity,
                    metapath_defs,
                )
                source_script = magnn_script
            write_channels(
                output_dir,
                channels,
                pipeline_name=pipeline_name,
                variants=variants,
                up_to_variant=up_to_variant,
                flavor=flavor,
                k=args.k,
                identity_mode=args.channel_identity,
                target_type=args.target_type,
                num_classes=args.num_classes,
                split_seed=args.split_seed,
                node_type_names=node_type_names,
                graph=graph,
                graph_audit=graph_audit,
                dataset_audit=dataset_audit,
                restricted_audit=restricted_audit,
                magnn_source_script=source_script,
            )
            print(f"Saved {flavor}: {output_dir} ({len(channels)} channels)")

    if reference_global_split is not None:
        print(
            f"Verified identical train/val/test global node IDs across all "
            f"generated datasets (split_seed={args.split_seed}, "
            f"reference={reference_split_name})."
        )


if __name__ == "__main__":
    main()
