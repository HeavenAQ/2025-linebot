from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from badminton_analysis.ml.models.skeleton_denoiser import SkeletonDenoiser
from badminton_analysis.ml.skeleton_dataset import (
    SkeletonCorrectionPairDataset,
    discover_sequence_files,
    load_sequence,
)
from badminton_analysis.ml.skeleton_normalization import phase_align_sequence
from badminton_analysis.ml.skeleton_normalization import restore_phase_timing
from badminton_analysis.ml.skeleton_scoring import (
    JOINT_WEIGHTS,
    correction_quality_metrics,
    expert_euclidean_distances,
    project_bone_lengths,
    select_bone_adapted_expert,
    sequence_training_losses,
)
from badminton_analysis.ml.skill_specs import get_skill_spec, supported_skill_choices


def _split_files(
    files: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    if len(files) < 10:
        raise ValueError("at least ten sequences are required for train/validation/test")
    shuffled = files.copy()
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.2))
    validation_count = max(1, round(len(shuffled) * 0.1))
    return (
        shuffled[test_count + validation_count :],
        shuffled[test_count : test_count + validation_count],
        shuffled[:test_count],
    )


def _split_expert_files(
    files: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_load_handedness(path), []).append(path)
    splits: list[list[Path]] = [[], [], []]
    for index, handedness in enumerate(sorted(grouped)):
        group = grouped[handedness]
        if len(group) < 3:
            raise ValueError(
                f"at least three {handedness}-handed experts are required"
            )
        shuffled = group.copy()
        random.Random(seed + index).shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * 0.2))
        validation_count = max(1, round(len(shuffled) * 0.1))
        group_splits = (
            shuffled[test_count + validation_count :],
            shuffled[test_count : test_count + validation_count],
            shuffled[:test_count],
        )
        for split, values in zip(splits, group_splits, strict=True):
            split.extend(values)
    for index, split in enumerate(splits):
        random.Random(seed + 100 + index).shuffle(split)
    return splits[0], splits[1], splits[2]


def _load_aligned(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sample = load_sequence(path)
    phases = sample["phase_indices"].astype(np.int64)
    skeleton = phase_align_sequence(sample["skeleton_3d"], phases)
    confidence = np.clip(
        phase_align_sequence(sample["confidence"], phases), 0.0, 1.0
    )
    return skeleton, confidence


def _load_expert_bank(
    files: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    samples = [_load_aligned(path) for path in files]
    return (
        np.stack([sample[0] for sample in samples]).astype(np.float32),
        np.stack([sample[1] for sample in samples]).astype(np.float32),
    )


def _load_expert_handedness(files: list[Path]) -> list[str]:
    return [_load_handedness(path) for path in files]


def _load_handedness(path: Path) -> str:
    with np.load(path, allow_pickle=False) as sample:
        if "handedness" not in sample:
            raise ValueError(f"dataset {path} has no handedness metadata")
        return str(sample["handedness"].item()).lower()


def _build_student_targets(
    student_files: list[Path],
    expert_files: list[Path],
    joint_weights: np.ndarray = JOINT_WEIGHTS,
    transition_weight: float = 0.0,
    transition_joints: tuple[int, ...] = (),
    transition_lean_joints: tuple[int, ...] = (),
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    expert_skeletons, expert_confidence = _load_expert_bank(expert_files)
    expert_handedness = np.asarray(_load_expert_handedness(expert_files))
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for path in student_files:
        source, source_confidence = _load_aligned(path)
        student_handedness = _load_handedness(path)
        allowed_indices = np.flatnonzero(expert_handedness == student_handedness)
        if not len(allowed_indices):
            raise ValueError(
                f"no {student_handedness}-handed expert is available for {path.name}"
            )
        local_index, target, target_confidence, selection_distance = (
            select_bone_adapted_expert(
                source,
                expert_skeletons[allowed_indices],
                source_confidence,
                expert_confidence[allowed_indices],
                joint_weights,
                transition_weight=transition_weight,
                transition_joints=transition_joints,
                transition_lean_joints=transition_lean_joints,
            )
        )
        nearest_index = int(allowed_indices[local_index])
        target_distance = expert_euclidean_distances(
            target,
            expert_skeletons[nearest_index : nearest_index + 1],
            target_confidence,
            expert_confidence[nearest_index : nearest_index + 1],
        )[0]
        targets[path.name] = (target, target_confidence)
        rows.append(
            {
                "student_file": path.name,
                "target_expert_file": expert_files[nearest_index].name,
                "input_target_expert_distance": selection_distance,
                "pseudo_target_expert_distance": float(target_distance),
            }
        )
    return targets, rows


def _expert_variability(
    expert_files: list[Path], quantile: float = 0.95
) -> dict[str, float]:
    """Measure natural expert variation without comparing a sample to itself."""
    if len(expert_files) < 2:
        raise ValueError("at least two experts are required to measure variability")
    skeletons, confidence = _load_expert_bank(expert_files)
    nearest_distances: list[float] = []
    for index in range(len(expert_files)):
        keep = np.arange(len(expert_files)) != index
        distances = expert_euclidean_distances(
            skeletons[index],
            skeletons[keep],
            confidence[index],
            confidence[keep],
        )
        nearest_distances.append(float(np.min(distances)))
    values = np.asarray(nearest_distances, dtype=np.float64)
    return {
        "count": float(len(values)),
        "mean_nearest_expert_distance": float(np.mean(values)),
        "p95_nearest_expert_distance": float(np.quantile(values, quantile)),
        "max_nearest_expert_distance": float(np.max(values)),
    }


def _run_epoch(
    model: SkeletonDenoiser,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    joint_weights: np.ndarray | None = None,
    transition_weight: float = 0.0,
    transition_joints: tuple[int, ...] = (),
    transition_lean_joints: tuple[int, ...] = (),
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        key: 0.0
        for key in (
            "loss",
            "position",
            "velocity",
            "angle",
            "bone_length",
            "transition",
        )
    }
    batches = 0
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        confidence = batch["confidence"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(features)
            losses = sequence_training_losses(
                prediction,
                target,
                confidence,
                JOINT_WEIGHTS if joint_weights is None else joint_weights,
                transition_weight=transition_weight,
                transition_joints=transition_joints,
                transition_lean_joints=transition_lean_joints,
            )
            if optimizer is not None:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def _evaluate_expert_distance(
    model: SkeletonDenoiser,
    student_files: list[Path],
    correction_expert_files: list[Path],
    expert_files: list[Path],
    device: torch.device,
    correction_strength: float,
    reference_guidance: float,
    expert_range_threshold: float,
    joint_weights: np.ndarray,
    transition_weight: float = 0.0,
    transition_joints: tuple[int, ...] = (),
    transition_lean_joints: tuple[int, ...] = (),
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    correction_expert_bank, correction_confidence_bank = _load_expert_bank(
        correction_expert_files
    )
    unseen_expert_bank, unseen_confidence_bank = _load_expert_bank(expert_files)
    correction_handedness = np.asarray(
        _load_expert_handedness(correction_expert_files)
    )
    unseen_handedness = np.asarray(_load_expert_handedness(expert_files))
    rows: list[dict[str, Any]] = []
    model.eval()
    for path in student_files:
        sample = load_sequence(path)
        student_handedness = str(sample["handedness"].item()).lower()
        correction_indices = np.flatnonzero(
            correction_handedness == student_handedness
        )
        unseen_indices = np.flatnonzero(unseen_handedness == student_handedness)
        if not len(correction_indices) or not len(unseen_indices):
            raise ValueError(
                f"same-handed expert validation data is unavailable for {path.name}"
            )
        correction_experts = correction_expert_bank[correction_indices]
        correction_expert_confidence = correction_confidence_bank[
            correction_indices
        ]
        unseen_experts = unseen_expert_bank[unseen_indices]
        unseen_expert_confidence = unseen_confidence_bank[unseen_indices]
        expert_skeletons = np.concatenate(
            (correction_experts, unseen_experts), axis=0
        )
        expert_confidence = np.concatenate(
            (correction_expert_confidence, unseen_expert_confidence), axis=0
        )
        allowed_correction_files = [
            correction_expert_files[int(index)] for index in correction_indices
        ]
        allowed_unseen_files = [expert_files[int(index)] for index in unseen_indices]
        all_expert_files = (*allowed_correction_files, *allowed_unseen_files)
        phases = sample["phase_indices"].astype(np.int64)
        raw_source = sample["skeleton_3d"].astype(np.float32)
        raw_confidence = sample["confidence"].astype(np.float32)
        source = phase_align_sequence(raw_source, phases)
        confidence = np.clip(
            phase_align_sequence(raw_confidence, phases), 0.0, 1.0
        )
        (
            correction_reference_index,
            correction_reference,
            _,
            _,
        ) = select_bone_adapted_expert(
            source,
            correction_experts,
            confidence,
            correction_expert_confidence,
            joint_weights,
            transition_weight=transition_weight,
            transition_joints=transition_joints,
            transition_lean_joints=transition_lean_joints,
        )
        features = np.concatenate(
            (source, correction_reference, confidence[..., None]), axis=-1
        )
        tensor = torch.as_tensor(
            features[None], dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            raw_prediction = model(tensor)[0].cpu().numpy()
        guided_prediction = (
            (1.0 - reference_guidance) * raw_prediction
            + reference_guidance * correction_reference
        )
        raw_prediction = source + correction_strength * (
            guided_prediction - source
        )
        corrected = project_bone_lengths(source, raw_prediction)
        input_distances = expert_euclidean_distances(
            source, expert_skeletons, confidence, expert_confidence
        )
        corrected_distances = expert_euclidean_distances(
            corrected, expert_skeletons, confidence, expert_confidence
        )
        corrected_unseen_distances = expert_euclidean_distances(
            corrected,
            unseen_experts,
            confidence,
            unseen_expert_confidence,
        )
        reference_target_distance = float(
            expert_euclidean_distances(
                corrected,
                correction_reference[None],
                confidence,
                np.minimum(
                    confidence,
                    correction_expert_confidence[correction_reference_index],
                )[None],
            )[0]
        )
        input_nearest = float(np.min(input_distances))
        corrected_nearest = float(np.min(corrected_distances))
        raw_correction = restore_phase_timing(corrected - source, phases)
        raw_corrected = project_bone_lengths(
            raw_source, raw_source + raw_correction
        )
        raw_reference_correction = restore_phase_timing(
            correction_reference - source, phases
        )
        raw_reference = project_bone_lengths(
            raw_source, raw_source + raw_reference_correction
        )
        quality = correction_quality_metrics(
            raw_source, raw_corrected, raw_confidence
        )
        reference_quality = correction_quality_metrics(
            raw_source, raw_reference, raw_confidence
        )
        reference_acceleration = reference_quality[
            "mean_correction_acceleration"
        ]
        rows.append(
            {
                "filename": path.name,
                "input_nearest_expert_distance": input_nearest,
                "corrected_nearest_expert_distance": corrected_nearest,
                "corrected_nearest_unseen_expert_distance": float(
                    np.min(corrected_unseen_distances)
                ),
                "corrected_reference_target_distance": reference_target_distance,
                "nearest_distance_improvement": input_nearest - corrected_nearest,
                "nearest_distance_ratio": corrected_nearest / max(input_nearest, 1e-8),
                "input_mean_expert_distance": float(np.mean(input_distances)),
                "corrected_mean_expert_distance": float(
                    np.mean(corrected_distances)
                ),
                "input_nearest_expert": all_expert_files[
                    int(np.argmin(input_distances))
                ].name,
                "corrected_nearest_expert": all_expert_files[
                    int(np.argmin(corrected_distances))
                ].name,
                "correction_reference_expert": correction_expert_files[
                    int(correction_indices[correction_reference_index])
                ].name,
                "within_expert_range": corrected_nearest
                <= expert_range_threshold,
                "reference_correction_acceleration": reference_acceleration,
                "correction_acceleration_ratio": quality[
                    "mean_correction_acceleration"
                ]
                / max(reference_acceleration, 1e-8),
                **quality,
            }
        )
    input_nearest_values = np.asarray(
        [row["input_nearest_expert_distance"] for row in rows], dtype=np.float64
    )
    corrected_nearest_values = np.asarray(
        [row["corrected_nearest_expert_distance"] for row in rows], dtype=np.float64
    )
    input_mean_values = np.asarray(
        [row["input_mean_expert_distance"] for row in rows], dtype=np.float64
    )
    corrected_mean_values = np.asarray(
        [row["corrected_mean_expert_distance"] for row in rows], dtype=np.float64
    )
    corrected_unseen_values = np.asarray(
        [
            row["corrected_nearest_unseen_expert_distance"]
            for row in rows
        ],
        dtype=np.float64,
    )
    reference_target_values = np.asarray(
        [row["corrected_reference_target_distance"] for row in rows],
        dtype=np.float64,
    )
    summary = {
        "count": float(len(rows)),
        "input_nearest_expert_distance": float(np.mean(input_nearest_values)),
        "corrected_nearest_expert_distance": float(
            np.mean(corrected_nearest_values)
        ),
        "corrected_nearest_unseen_expert_distance": float(
            np.mean(corrected_unseen_values)
        ),
        "nearest_distance_ratio": float(
            np.mean(corrected_nearest_values / np.maximum(input_nearest_values, 1e-8))
        ),
        "improved_fraction": float(
            np.mean(corrected_nearest_values < input_nearest_values)
        ),
        "expert_range_threshold": expert_range_threshold,
        "within_expert_range_fraction": float(
            np.mean(corrected_nearest_values <= expert_range_threshold)
        ),
        "mean_reference_target_distance": float(
            np.mean(reference_target_values)
        ),
        "max_reference_target_distance": float(np.max(reference_target_values)),
        "input_mean_expert_distance": float(np.mean(input_mean_values)),
        "corrected_mean_expert_distance": float(np.mean(corrected_mean_values)),
        "mean_expert_distance_ratio": float(
            np.mean(corrected_mean_values / np.maximum(input_mean_values, 1e-8))
        ),
        "max_joint_correction": float(
            max(row["max_joint_correction"] for row in rows)
        ),
        "mean_correction_acceleration": float(
            np.mean([row["mean_correction_acceleration"] for row in rows])
        ),
        "mean_reference_correction_acceleration": float(
            np.mean(
                [row["reference_correction_acceleration"] for row in rows]
            )
        ),
        "mean_correction_acceleration_ratio": float(
            np.mean([row["correction_acceleration_ratio"] for row in rows])
        ),
        "p95_relative_bone_change": float(
            max(row["p95_relative_bone_change"] for row in rows)
        ),
    }
    return summary, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a phase-aligned, expert-guided skeleton corrector"
    )
    parser.add_argument(
        "--skill", choices=supported_skill_choices(), default="clear"
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--model-path")
    parser.add_argument("--metrics-dir")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--max-correction", type=float, default=3.5)
    parser.add_argument("--reference-guidance", type=float, default=0.5)
    parser.add_argument("--minimum-improved-fraction", type=float, default=1.0)
    parser.add_argument("--minimum-within-expert-range", type=float, default=1.0)
    parser.add_argument("--maximum-reference-distance", type=float, default=0.1)
    parser.add_argument(
        "--maximum-correction-acceleration-ratio", type=float, default=1.1
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.reference_guidance <= 1.0:
        raise ValueError("reference guidance must be between zero and one")
    spec = get_skill_spec(args.skill)
    dataset_root = Path(args.dataset_root) if args.dataset_root else spec.dataset_root
    model_path = Path(args.model_path) if args.model_path else spec.model_path
    metrics_dir = (
        Path(args.metrics_dir)
        if args.metrics_dir
        else spec.training_metrics_dir
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    expert_files = discover_sequence_files(dataset_root, "experts")
    student_files = discover_sequence_files(dataset_root, "beginners")
    for path in (*expert_files, *student_files):
        sample_skill = str(load_sequence(path)["skill"].item())
        if sample_skill != spec.slug:
            raise ValueError(
                f"dataset {path} contains skill {sample_skill}, expected {spec.slug}"
            )
    expert_train, expert_validation, expert_test = _split_expert_files(
        expert_files, args.seed
    )
    student_train, student_validation, student_test = _split_files(
        student_files, args.seed + 1
    )
    student_targets, pairing_rows = _build_student_targets(
        student_train,
        expert_train,
        spec.joint_weights_array,
        spec.transition_weight,
        spec.transition_joints,
        spec.transition_lean_joints,
    )
    validation_expert_variability = _expert_variability(expert_validation)
    validation_expert_threshold = validation_expert_variability[
        "p95_nearest_expert_distance"
    ]
    test_expert_variability = _expert_variability(expert_test)
    test_expert_threshold = test_expert_variability[
        "p95_nearest_expert_distance"
    ]

    expert_training_dataset = SkeletonCorrectionPairDataset(
        expert_train, reference_conditioned=True, augment=True
    )
    student_training_dataset = SkeletonCorrectionPairDataset(
        student_train,
        targets=student_targets,
        reference_conditioned=True,
        augment=True,
    )
    expert_validation_dataset = SkeletonCorrectionPairDataset(
        expert_validation,
        reference_conditioned=True,
        augment=True,
        deterministic=True,
    )
    training_loader: DataLoader[Any] = DataLoader(
        ConcatDataset((expert_training_dataset, student_training_dataset)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader: DataLoader[Any] = DataLoader(
        expert_validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SkeletonDenoiser(
        input_features=7,
        model_dim=args.model_dim,
        max_correction=args.max_correction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_dir / "training_pairs.csv", pairing_rows)

    rows: list[dict[str, float | int]] = []
    best_expert_distance = float("inf")
    accepted_epoch: int | None = None
    for epoch in range(1, args.epochs + 1):
        training_metrics = _run_epoch(
            model,
            training_loader,
            device,
            optimizer,
            spec.joint_weights_array,
            spec.transition_weight,
            spec.transition_joints,
            spec.transition_lean_joints,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                None,
                spec.joint_weights_array,
                spec.transition_weight,
                spec.transition_joints,
                spec.transition_lean_joints,
            )
        guidance, _ = _evaluate_expert_distance(
            model,
            student_validation,
            expert_train,
            expert_validation,
            device,
            1.0,
            args.reference_guidance,
            validation_expert_threshold,
            spec.joint_weights_array,
            spec.transition_weight,
            spec.transition_joints,
            spec.transition_lean_joints,
        )
        if not all(
            np.isfinite(value)
            for value in (
                training_metrics["loss"],
                validation_metrics["loss"],
                guidance["corrected_nearest_expert_distance"],
            )
        ):
            raise RuntimeError("training produced a non-finite metric")
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        row.update({f"train_{key}": value for key, value in training_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        row.update({f"guidance_{key}": value for key, value in guidance.items()})
        rows.append(row)
        _write_csv(metrics_dir / "training_metrics.csv", rows)

        accepted = (
            guidance["improved_fraction"] >= args.minimum_improved_fraction
            and guidance["within_expert_range_fraction"]
            >= args.minimum_within_expert_range
            and guidance["corrected_nearest_expert_distance"]
            <= validation_expert_threshold
            and guidance["max_reference_target_distance"]
            <= args.maximum_reference_distance
            and guidance["max_joint_correction"] <= args.max_correction * 1.5
            and guidance["mean_correction_acceleration_ratio"]
            <= args.maximum_correction_acceleration_ratio
            and guidance["p95_relative_bone_change"] < 1e-3
        )
        if (
            accepted
            and guidance["corrected_nearest_expert_distance"]
            < best_expert_distance
        ):
            best_expert_distance = guidance["corrected_nearest_expert_distance"]
            accepted_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "skill": spec.slug,
                    "joint_weights": list(spec.joint_weights),
                    "transition_weight": spec.transition_weight,
                    "transition_joints": list(spec.transition_joints),
                    "transition_lean_joints": list(spec.transition_lean_joints),
                    "criteria": [rule.as_prompt_dict() for rule in spec.rules],
                    "sequence_frames": 64,
                    "phase_aligned": True,
                    "expert_guided": True,
                    "reference_conditioned": True,
                    "target_definition": "full_bone_preserving_nearest_expert",
                    "inference_strength": 1.0,
                    "reference_guidance": args.reference_guidance,
                    "quality_gates": {
                        "minimum_improved_fraction": args.minimum_improved_fraction,
                        "minimum_within_expert_range": args.minimum_within_expert_range,
                        "maximum_reference_distance": args.maximum_reference_distance,
                        "maximum_correction_acceleration_ratio": (
                            args.maximum_correction_acceleration_ratio
                        ),
                    },
                    "accepted_epoch": epoch,
                    "validation_expert_distance": guidance,
                    "validation_expert_variability": validation_expert_variability,
                    "expert_training_files": [path.name for path in expert_train],
                    "expert_reference_handedness": _load_expert_handedness(
                        expert_train
                    ),
                    "expert_reference_skeletons": _load_expert_bank(expert_train)[0],
                    "expert_reference_confidence": _load_expert_bank(expert_train)[1],
                    "expert_validation_files": [
                        path.name for path in expert_validation
                    ],
                    "expert_test_files": [path.name for path in expert_test],
                    "student_training_files": [path.name for path in student_train],
                    "student_validation_files": [
                        path.name for path in student_validation
                    ],
                    "student_test_files": [path.name for path in student_test],
                },
                model_path,
            )
        scheduler.step()
        print(
            f"epoch={epoch:03d} train={training_metrics['loss']:.6f} "
            f"expert_val={validation_metrics['loss']:.6f} "
            f"distance={guidance['corrected_nearest_expert_distance']:.6f} "
            f"ratio={guidance['nearest_distance_ratio']:.4f} "
            f"acceleration_ratio={guidance['mean_correction_acceleration_ratio']:.4f} "
            f"improved={guidance['improved_fraction']:.2f} "
            f"within_expert_range={guidance['within_expert_range_fraction']:.2f} "
            f"accepted={accepted}"
        )

    if accepted_epoch is None:
        raise RuntimeError(
            "no checkpoint satisfied the expert-distance and quality criteria"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_summary, validation_rows = _evaluate_expert_distance(
        model,
        student_validation,
        expert_train,
        expert_validation,
        device,
        1.0,
        args.reference_guidance,
        validation_expert_threshold,
        spec.joint_weights_array,
        spec.transition_weight,
        spec.transition_joints,
        spec.transition_lean_joints,
    )
    test_summary, test_rows = _evaluate_expert_distance(
        model,
        student_test,
        expert_train,
        expert_test,
        device,
        1.0,
        args.reference_guidance,
        test_expert_threshold,
        spec.joint_weights_array,
        spec.transition_weight,
        spec.transition_joints,
        spec.transition_lean_joints,
    )
    all_students_summary, _ = _evaluate_expert_distance(
        model,
        student_files,
        expert_train,
        expert_test,
        device,
        1.0,
        args.reference_guidance,
        test_expert_threshold,
        spec.joint_weights_array,
        spec.transition_weight,
        spec.transition_joints,
        spec.transition_lean_joints,
    )
    _write_csv(metrics_dir / "expert_distance_validation.csv", validation_rows)
    _write_csv(metrics_dir / "expert_distance_test.csv", test_rows)
    summary = {
        "accepted_epoch": accepted_epoch,
        "validation_expert_variability": validation_expert_variability,
        "test_expert_variability": test_expert_variability,
        "validation": validation_summary,
        "test": test_summary,
        "all_students_quality": {
            "count": all_students_summary["count"],
            "max_joint_correction": all_students_summary[
                "max_joint_correction"
            ],
            "mean_correction_acceleration": all_students_summary[
                "mean_correction_acceleration"
            ],
            "mean_reference_correction_acceleration": all_students_summary[
                "mean_reference_correction_acceleration"
            ],
            "mean_correction_acceleration_ratio": all_students_summary[
                "mean_correction_acceleration_ratio"
            ],
            "p95_relative_bone_change": all_students_summary[
                "p95_relative_bone_change"
            ],
            "max_reference_target_distance": all_students_summary[
                "max_reference_target_distance"
            ],
        },
    }
    if (
        all_students_summary["max_joint_correction"]
        > args.max_correction * 1.5
        or all_students_summary["mean_correction_acceleration_ratio"]
        > args.maximum_correction_acceleration_ratio
        or all_students_summary["p95_relative_bone_change"] >= 1e-3
        or all_students_summary["max_reference_target_distance"]
        > args.maximum_reference_distance
    ):
        raise RuntimeError("accepted checkpoint failed the all-student safety audit")
    (metrics_dir / "expert_distance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    checkpoint["test_expert_distance"] = test_summary
    torch.save(checkpoint, model_path)
    print(f"Saved accepted checkpoint: {model_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
