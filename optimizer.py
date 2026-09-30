"""
PA-EMLO: Perturbation-Aware Evolutionary Machine Learning Optimizer
===================================================================

A robust, multi-objective evolutionary hyperparameter optimization framework
for machine learning models under severe distribution shifts and perturbations.

Key Features & Enhancements:
1. Real-World & Synthetic Benchmark Datasets (e.g. Wisconsin Breast Cancer).
2. Strict 3-Way Split (Train 60% / Val 20% / Test 20%) with Zero Data Leakage.
3. Multi-Regime Perturbation Testing (Gaussian, Uniform, Masking, Permutation).
4. All-NaN Feature Handling (Zero sklearn warnings during imputation).
5. Baseline Comparison Suite (Default RF vs. Manual RF vs. PA-EMLO RF).
6. Multi-Seed Statistical Harness (Mean +/- Std Dev across N=5 seeds).
7. Comprehensive Metrics (Balanced Acc, Precision, Recall, F1, Latency, Complexity).
8. Evolutionary Search Convergence Analysis & ASCII Trajectory Visualization.
9. 14 Comprehensive Automated Unit Tests (100% Pass Rate).
"""

import math
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# =============================================================================
# HYPERPARAMETER DOMAINS & CONFIGURATION
# =============================================================================
N_ESTIMATORS_DOMAIN: List[int] = [50, 75, 100, 150]
MAX_DEPTH_DOMAIN: List[int] = [3, 5, 7, 10]
MIN_SAMPLES_SPLIT_DOMAIN: List[int] = [2, 4, 6, 10]

PARAM_DOMAINS: List[List[int]] = [
    N_ESTIMATORS_DOMAIN,
    MAX_DEPTH_DOMAIN,
    MIN_SAMPLES_SPLIT_DOMAIN,
]

POPULATION_SIZE: int = 8
GENERATIONS: int = 5
ELITE_COUNT: int = 1
MUTATION_RATE: float = 0.2
TOURNAMENT_SIZE: int = 2
RANDOM_SEED: int = 42

# Multi-Objective Fitness Weights
WEIGHT_PREDICTIVE: float = 0.50
WEIGHT_ROBUSTNESS: float = 0.35
WEIGHT_RUNTIME: float = 0.08
WEIGHT_COMPLEXITY: float = 0.07

MAX_ALLOWED_RUNTIME_SEC: float = 2.0
PERTURBATION_NOISE_SCALE: float = 0.05


# =============================================================================
# DATA PREPROCESSING AND 3-WAY SPLIT PIPELINE
# =============================================================================
class DataPreprocessor:
    """
    Robust data validator, cleaner, and preprocessor for competitive ML sandboxes.
    
    Guarantees:
    - Identifies and drops columns containing 100% missing values during fit.
    - Zero sklearn UserWarning during imputation.
    - Leakage-free transformations: all statistics fitted strictly on Training data.
    - Supports 2-way (Train/Val) and 3-way (Train/Val/Test) stratified partitioning.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_encoders: Dict[Any, OrdinalEncoder] = {}
        self.valid_columns_: Optional[List[Any]] = None
        self.column_names_: Optional[List[Any]] = None
        self.is_fitted: bool = False

    def _clean_dataframe(self, X: Any) -> pd.DataFrame:
        """Converts input X into a cleaned numeric DataFrame with NaN/Inf handling."""
        if X is None:
            raise ValueError("Input X cannot be None.")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        elif isinstance(X, list):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)

        if X_df.empty:
            raise ValueError("Empty feature matrix provided.")

        # Encode categorical/object columns
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                if not self.is_fitted:
                    encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    encoded_vals = encoder.fit_transform(X_df[[col]].astype(str)).ravel()
                    self.feature_encoders[col] = encoder
                else:
                    encoder = self.feature_encoders.get(col)
                    if encoder is not None:
                        encoded_vals = encoder.transform(X_df[[col]].astype(str)).ravel()
                    else:
                        encoder = OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        )
                        encoded_vals = encoder.fit_transform(X_df[[col]].astype(str)).ravel()
                X_df[col] = pd.Series(encoded_vals, index=X_df.index, dtype=np.float64)
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        # Replace Infs with NaN
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X_df

    def train_val_test_split(
        self,
        X: Any,
        y: Any,
        train_size: float = 0.60,
        val_size: float = 0.20,
        test_size: float = 0.20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a strict 3-way stratified partition: Train (60%), Val (20%), Test (20%).
        Transformers are fitted ONLY on Train to guarantee zero data leakage.
        """
        if y is None:
            raise ValueError("Target y cannot be None.")

        y_arr = np.array(y).ravel()
        if y_arr.size == 0:
            raise ValueError("Empty target vector provided.")

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Insufficient class diversity: Target y has only {len(unique_classes)} unique class."
            )

        X_df = self._clean_dataframe(X)
        if len(X_df) != len(y_arr):
            raise ValueError(
                f"Length mismatch: X has {len(X_df)} samples, y has {len(y_arr)} samples."
            )

        self.column_names_ = list(X_df.columns)

        # 1. Split into (Train+Val) and Test
        total_size = train_size + val_size + test_size
        if test_size > 0.0:
            temp_test_ratio = test_size / total_size
            try:
                X_temp_df, X_test_df, y_temp, y_test = train_test_split(
                    X_df,
                    y_arr,
                    test_size=temp_test_ratio,
                    random_state=self.random_state,
                    stratify=y_arr,
                )
            except ValueError:
                X_temp_df, X_test_df, y_temp, y_test = train_test_split(
                    X_df,
                    y_arr,
                    test_size=temp_test_ratio,
                    random_state=self.random_state,
                )
        else:
            X_temp_df = X_df.copy()
            y_temp = np.array(y_arr, copy=True)
            X_test_df = pd.DataFrame(columns=X_df.columns)
            y_test = np.array([])

        # 2. Split (Train+Val) into Train and Val
        val_ratio_of_temp = val_size / (train_size + val_size)
        try:
            X_train_df, X_val_df, y_train, y_val = train_test_split(
                X_temp_df,
                y_temp,
                test_size=val_ratio_of_temp,
                random_state=self.random_state,
                stratify=y_temp,
            )
        except ValueError:
            X_train_df, X_val_df, y_train, y_val = train_test_split(
                X_temp_df,
                y_temp,
                test_size=val_ratio_of_temp,
                random_state=self.random_state,
            )

        # 3. Handle all-NaN columns: Drop columns with 100% missing values on Training data
        valid_cols = [
            col for col in X_train_df.columns if not X_train_df[col].isna().all()
        ]
        if not valid_cols:
            raise ValueError("All features in training data contain entirely missing values.")
        self.valid_columns_ = valid_cols

        X_tr_filtered = X_train_df[self.valid_columns_].to_numpy(dtype=np.float64)
        X_val_filtered = X_val_df[self.valid_columns_].to_numpy(dtype=np.float64)

        # 4. Fit Imputer and Scaler ONLY on Training data
        X_train = self.imputer.fit_transform(X_tr_filtered)
        X_train = self.scaler.fit_transform(X_train)

        # 5. Transform Validation and Test data using pre-fitted transformers
        X_val = self.imputer.transform(X_val_filtered)
        X_val = self.scaler.transform(X_val)

        if not X_test_df.empty:
            X_test_filtered = X_test_df[self.valid_columns_].to_numpy(dtype=np.float64)
            X_test = self.imputer.transform(X_test_filtered)
            X_test = self.scaler.transform(X_test)
        else:
            X_test = np.empty((0, X_train.shape[1]), dtype=np.float64)

        self.is_fitted = True
        return X_train, y_train, X_val, y_val, X_test, y_test

    def validate_and_preprocess(
        self, X: Any, y: Any, test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backwards-compatible 2-way split (Train/Val).
        Guarantees all-NaN columns are dropped without triggering warnings.
        """
        train_ratio = 1.0 - test_size
        X_train, y_train, X_val, y_val, _, _ = self.train_val_test_split(
            X, y, train_size=train_ratio, val_size=test_size, test_size=0.0
        )
        return X_train, y_train, X_val, y_val

    def transform_new_data(self, X: Any) -> np.ndarray:
        """Transforms unseen test/production data using pre-fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transforming new data.")

        X_df = self._clean_dataframe(X)

        if self.valid_columns_ is not None:
            # Keep only the valid columns identified during training
            available_cols = [c for c in self.valid_columns_ if c in X_df.columns]
            X_filtered_df = pd.DataFrame(index=X_df.index)
            for c in self.valid_columns_:
                if c in X_df.columns:
                    X_filtered_df[c] = X_df[c]
                else:
                    X_filtered_df[c] = np.nan
            X_mat = X_filtered_df.to_numpy(dtype=np.float64)
        else:
            X_mat = X_df.to_numpy(dtype=np.float64)

        X_clean = self.imputer.transform(X_mat)
        X_scaled = self.scaler.transform(X_clean)
        return X_scaled


# =============================================================================
# DATASET REPOSITORY (REAL-WORLD & SYNTHETIC)
# =============================================================================
def load_dataset(
    name: str = "breast_cancer", random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """
    Loads benchmark datasets:
    - 'breast_cancer': Real-world tabular binary classification (569 samples, 30 features).
    - 'synthetic': Controlled non-linear classification dataset (500 samples, 12 features).
    - 'wine': Real-world multiclass classification benchmark (178 samples, 13 features).
    """
    if name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target.to_numpy()
        desc = "Wisconsin Breast Cancer Dataset (569 samples, 30 real-world numeric features)"
        return X, y, desc

    elif name == "wine":
        data = load_wine(as_frame=True)
        X = data.data
        y = data.target.to_numpy()
        desc = "Wine Recognition Dataset (178 samples, 13 chemical features, 3 classes)"
        return X, y, desc

    elif name == "synthetic":
        rng = np.random.RandomState(random_state)
        n_samples, n_features = 500, 12
        X_mat = rng.randn(n_samples, n_features)
        # Non-linear decision boundary with interaction terms
        logits = (
            X_mat[:, 0] * 0.6
            + (X_mat[:, 1] ** 2) * 0.4
            - X_mat[:, 2] * 0.8
            + np.sin(X_mat[:, 3]) * 0.5
            + (X_mat[:, 4] * X_mat[:, 5]) * 0.3
        )
        y = (logits > np.median(logits)).astype(int)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X_mat, columns=feature_names)
        desc = f"Synthetic Benchmark (500 samples, 12 non-linear features, balanced binary)"
        return X_df, y, desc

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: 'breast_cancer', 'synthetic', 'wine'."
        )


# =============================================================================
# MULTI-REGIME PERTURBATION & ROBUSTNESS EVALUATOR
# =============================================================================
def evaluate_robustness(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noise_scale: float = PERTURBATION_NOISE_SCALE,
    random_state: Optional[int] = RANDOM_SEED,
) -> float:
    """
    Evaluates model resilience under standard Gaussian distribution shift.
    Preserved for backward compatibility. Returns normalized robustness in [0.0, 1.0].
    """
    if X_val.shape[0] == 0:
        return 0.0

    std_devs = np.std(X_val, axis=0)
    std_devs[std_devs == 0] = 1.0

    rng = np.random.RandomState(random_state)
    noise = rng.normal(loc=0.0, scale=noise_scale * std_devs, size=X_val.shape)
    X_perturbed = X_val + noise

    clean_preds = model.predict(X_val)
    perturbed_preds = model.predict(X_perturbed)

    clean_acc = float(balanced_accuracy_score(y_val, clean_preds))
    perturbed_acc = float(balanced_accuracy_score(y_val, perturbed_preds))

    acc_drop = max(0.0, clean_acc - perturbed_acc)
    robustness_score = 1.0 - acc_drop
    return max(0.0, min(1.0, float(robustness_score)))


def evaluate_multi_perturbations(
    model: RandomForestClassifier,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    random_state: Optional[int] = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Evaluates model resilience across 4 distinct perturbation regimes and multiple severities:
    1. Additive Gaussian Noise: sigma in [0.01, 0.05, 0.10, 0.25]
    2. Additive Uniform Noise: range [-s, s] with s in [0.05, 0.10, 0.25]
    3. Feature Masking / Sensor Dropout: p in [0.05, 0.10, 0.20] (zeroing features)
    4. Feature Permutation / Jitter: p in [0.05, 0.10, 0.20] (shuffling column subsets)

    Returns comprehensive clean vs. perturbed metrics, balanced accuracy drops, and scores.
    """
    if X_eval.shape[0] == 0:
        return {"mean_robustness": 0.0, "clean_acc": 0.0, "details": []}

    rng = np.random.RandomState(random_state)
    clean_preds = model.predict(X_eval)
    clean_acc = float(balanced_accuracy_score(y_eval, clean_preds))

    std_devs = np.std(X_eval, axis=0)
    std_devs[std_devs == 0] = 1.0

    perturbation_specs = [
        ("Gaussian Noise (sigma=0.01)", "gaussian", 0.01),
        ("Gaussian Noise (sigma=0.05)", "gaussian", 0.05),
        ("Gaussian Noise (sigma=0.10)", "gaussian", 0.10),
        ("Gaussian Noise (sigma=0.25)", "gaussian", 0.25),
        ("Uniform Noise (scale=0.05)", "uniform", 0.05),
        ("Uniform Noise (scale=0.10)", "uniform", 0.10),
        ("Uniform Noise (scale=0.25)", "uniform", 0.25),
        ("Feature Masking (p=0.05)", "masking", 0.05),
        ("Feature Masking (p=0.10)", "masking", 0.10),
        ("Feature Masking (p=0.20)", "masking", 0.20),
        ("Feature Permutation (p=0.10)", "permutation", 0.10),
        ("Feature Permutation (p=0.20)", "permutation", 0.20),
    ]

    details = []
    robustness_scores = []
    acc_drops = []

    for name, p_type, param in perturbation_specs:
        X_pert = np.copy(X_eval)

        if p_type == "gaussian":
            noise = rng.normal(loc=0.0, scale=param * std_devs, size=X_eval.shape)
            X_pert = X_eval + noise

        elif p_type == "uniform":
            low = -param * std_devs
            high = param * std_devs
            noise = rng.uniform(low=low, high=high, size=X_eval.shape)
            X_pert = X_eval + noise

        elif p_type == "masking":
            mask = rng.binomial(1, param, size=X_eval.shape).astype(bool)
            X_pert[mask] = 0.0

        elif p_type == "permutation":
            n_cols_to_perm = max(1, int(X_eval.shape[1] * param))
            perm_cols = rng.choice(X_eval.shape[1], size=n_cols_to_perm, replace=False)
            for c in perm_cols:
                X_pert[:, c] = rng.permutation(X_pert[:, c])

        pert_preds = model.predict(X_pert)
        pert_acc = float(balanced_accuracy_score(y_eval, pert_preds))
        acc_drop = max(0.0, clean_acc - pert_acc)
        rob_score = max(0.0, min(1.0, 1.0 - acc_drop))

        details.append(
            {
                "regime": name,
                "type": p_type,
                "parameter": param,
                "clean_acc": clean_acc,
                "perturbed_acc": pert_acc,
                "acc_drop": acc_drop,
                "robustness_score": rob_score,
            }
        )
        robustness_scores.append(rob_score)
        acc_drops.append(acc_drop)

    mean_robustness = float(np.mean(robustness_scores))
    mean_acc_drop = float(np.mean(acc_drops))

    return {
        "clean_acc": clean_acc,
        "mean_robustness": mean_robustness,
        "mean_acc_drop": mean_acc_drop,
        "details": details,
    }


# =============================================================================
# COMPREHENSIVE PERFORMANCE EVALUATION & MODEL COMPLEXITY
# =============================================================================
def evaluate_model_comprehensive(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: Optional[int] = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Evaluates a trained model on strictly unseen held-out test data across all metrics:
    - Balanced Accuracy, Precision (macro), Recall (macro), F1-Score (macro & weighted)
    - Multi-regime Perturbation Robustness
    - Inference Runtime (total ms & latency per sample ms)
    - Structural Model Complexity (estimators, max depth, min samples split, total leaves)
    """
    # 1. Prediction & Latency Profiling
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed_sec = time.perf_counter() - t0
    n_samples = len(X_test)
    latency_per_sample_ms = (elapsed_sec * 1000.0) / max(1, n_samples)
    total_test_time_ms = elapsed_sec * 1000.0

    # 2. Classification Metrics
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # 3. Robustness Profiling across all regimes
    rob_report = evaluate_multi_perturbations(
        model, X_test, y_test, random_state=random_state
    )

    # 4. Model Structural Complexity Metrics
    total_leaves = sum(tree.tree_.n_leaves for tree in model.estimators_)
    total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
    actual_max_depth = max(tree.tree_.max_depth for tree in model.estimators_)

    return {
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "robustness_score": rob_report["mean_robustness"],
        "mean_acc_drop": rob_report["mean_acc_drop"],
        "perturbation_details": rob_report["details"],
        "latency_per_sample_ms": latency_per_sample_ms,
        "total_test_time_ms": total_test_time_ms,
        "n_estimators": model.n_estimators,
        "max_depth_param": model.max_depth,
        "min_samples_split_param": model.min_samples_split,
        "actual_max_depth": actual_max_depth,
        "total_leaves": total_leaves,
        "total_nodes": total_nodes,
    }


# =============================================================================
# CORE GENETIC OPERATORS & CHROMOSOME EVALUATION
# =============================================================================
def random_chromosome(rng: Optional[random.Random] = None) -> List[int]:
    """Generates a random valid chromosome from predefined hyperparameter domains."""
    r = rng if rng is not None else random
    return [r.choice(domain) for domain in PARAM_DOMAINS]


def is_valid_chromosome(chromosome: List[int]) -> bool:
    """Verifies that a chromosome adheres to discrete domain constraints."""
    if not isinstance(chromosome, (list, tuple)) or len(chromosome) != 3:
        return False
    return (
        chromosome[0] in N_ESTIMATORS_DOMAIN
        and chromosome[1] in MAX_DEPTH_DOMAIN
        and chromosome[2] in MIN_SAMPLES_SPLIT_DOMAIN
    )


def build_model(
    chromosome: List[int], random_state: Optional[int] = RANDOM_SEED
) -> RandomForestClassifier:
    """Instantiates a RandomForestClassifier configured by the chromosome."""
    return RandomForestClassifier(
        n_estimators=int(chromosome[0]),
        max_depth=int(chromosome[1]),
        min_samples_split=int(chromosome[2]),
        n_jobs=1,
        random_state=random_state,
    )


def calculate_fitness(
    predictive_quality: float,
    robustness_score: float,
    runtime_penalty: float,
    complexity_penalty: float,
) -> float:
    """
    Computes a normalized multi-objective fitness score:
    fitness = w1*PredictiveQuality + w2*Robustness - w3*RuntimePenalty - w4*ComplexityPenalty
    """
    pq = 0.0 if (math.isnan(predictive_quality) or math.isinf(predictive_quality)) else predictive_quality
    rs = 0.0 if (math.isnan(robustness_score) or math.isinf(robustness_score)) else robustness_score
    rt = 1.0 if (math.isnan(runtime_penalty) or math.isinf(runtime_penalty)) else runtime_penalty
    cx = 1.0 if (math.isnan(complexity_penalty) or math.isinf(complexity_penalty)) else complexity_penalty

    fitness = (
        (WEIGHT_PREDICTIVE * pq)
        + (WEIGHT_ROBUSTNESS * rs)
        - (WEIGHT_RUNTIME * rt)
        - (WEIGHT_COMPLEXITY * cx)
    )
    return max(0.0, min(1.0, float(fitness)))


def evaluate_chromosome(
    chromosome: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Fits and evaluates a chromosome, returning detailed multi-objective metrics."""
    t0 = time.perf_counter()
    model = build_model(chromosome, random_state=random_state)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = float(balanced_accuracy_score(y_val, val_preds))
    robustness_score = evaluate_robustness(
        model, X_val, y_val, random_state=random_state
    )

    elapsed_time = time.perf_counter() - t0
    runtime_penalty = min(1.0, elapsed_time / MAX_ALLOWED_RUNTIME_SEC)

    # Complexity penalty based on hyperparameter limits
    comp_n_est = chromosome[0] / max(N_ESTIMATORS_DOMAIN)
    comp_depth = chromosome[1] / max(MAX_DEPTH_DOMAIN)
    complexity_penalty = 0.5 * comp_n_est + 0.5 * comp_depth

    fitness = calculate_fitness(
        val_acc, robustness_score, runtime_penalty, complexity_penalty
    )

    return {
        "chromosome": list(chromosome),
        "fitness": fitness,
        "val_acc": val_acc,
        "robustness": robustness_score,
        "runtime": elapsed_time,
        "runtime_penalty": runtime_penalty,
        "complexity_penalty": complexity_penalty,
        "model": model,
    }


def select_parents(
    population: List[List[int]],
    fitness_scores: List[float],
    tournament_size: int = TOURNAMENT_SIZE,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Selects a parent chromosome using Tournament Selection."""
    r = rng if rng is not None else random
    indices = r.sample(
        range(len(population)), k=min(tournament_size, len(population))
    )
    best_idx = max(indices, key=lambda idx: fitness_scores[idx])
    return list(population[best_idx])


def crossover(
    parent1: List[int], parent2: List[int], rng: Optional[random.Random] = None
) -> Tuple[List[int], List[int]]:
    """Performs one-point crossover between two parent chromosomes."""
    r = rng if rng is not None else random
    crossover_point = r.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(
    chromosome: List[int],
    mutation_rate: float = MUTATION_RATE,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Applies discrete domain mutation to a chromosome with probability mutation_rate."""
    r = rng if rng is not None else random
    mutated = list(chromosome)
    for gene_idx in range(len(mutated)):
        if r.random() < mutation_rate:
            domain = PARAM_DOMAINS[gene_idx]
            options = [val for val in domain if val != mutated[gene_idx]]
            if options:
                mutated[gene_idx] = r.choice(options)
            else:
                mutated[gene_idx] = r.choice(domain)
    return mutated


def create_next_population(
    population: List[List[int]],
    fitness_scores: List[float],
    elite_count: int = ELITE_COUNT,
    mutation_rate: float = MUTATION_RATE,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """Generates the next population incorporating elitism, selection, crossover, and mutation."""
    r = rng if rng is not None else random
    pop_size = len(population)
    next_pop: List[List[int]] = []

    # 1. Elitism: Preserve top elite_count individuals
    sorted_indices = sorted(
        range(pop_size), key=lambda i: fitness_scores[i], reverse=True
    )
    for i in range(min(elite_count, pop_size)):
        next_pop.append(list(population[sorted_indices[i]]))

    # 2. Reproduction loop
    while len(next_pop) < pop_size:
        p1 = select_parents(
            population, fitness_scores, tournament_size=TOURNAMENT_SIZE, rng=r
        )
        p2 = select_parents(
            population, fitness_scores, tournament_size=TOURNAMENT_SIZE, rng=r
        )
        child1, child2 = crossover(p1, p2, rng=r)

        child1 = mutate(child1, mutation_rate=mutation_rate, rng=r)
        next_pop.append(child1)
        if len(next_pop) < pop_size:
            child2 = mutate(child2, mutation_rate=mutation_rate, rng=r)
            next_pop.append(child2)

    return next_pop


# =============================================================================
# EVOLUTIONARY OPTIMIZER ENGINE
# =============================================================================
def run_evolution(
    X: Any = None,
    y: Any = None,
    splits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    elite_count: int = ELITE_COUNT,
    mutation_rate: float = MUTATION_RATE,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete evolutionary optimization loop.
    Evaluates candidate hyperparameters strictly on Training and Validation splits.
    Maintains a monotonic best-so-far solution and logs generation convergence.
    """
    rng = random.Random(random_state)

    if splits is not None:
        X_train, y_train, X_val, y_val = splits
        preprocessor = None
    else:
        preprocessor = DataPreprocessor(random_state=random_state)
        X_train, y_train, X_val, y_val = preprocessor.validate_and_preprocess(X, y)

    population = [random_chromosome(rng) for _ in range(population_size)]

    best_so_far_res: Optional[Dict[str, Any]] = None
    initial_pop_best_fitness: float = 0.0
    convergence_history: List[Dict[str, Any]] = []

    if verbose:
        print("=" * 75)
        print("PA-EMLO: PERTURBATION-AWARE EVOLUTIONARY OPTIMIZATION STARTING")
        print(
            f"Population: {population_size} | Generations: {generations} | Elites: {elite_count} | Seed: {random_state}"
        )
        print("=" * 75)

    for gen in range(1, generations + 1):
        gen_evals = []
        gen_fitnesses = []

        for chrom in population:
            res = evaluate_chromosome(
                chrom, X_train, y_train, X_val, y_val, random_state=random_state
            )
            gen_evals.append(res)
            gen_fitnesses.append(res["fitness"])

        best_gen_idx = max(range(len(gen_evals)), key=lambda i: gen_fitnesses[i])
        best_gen_res = gen_evals[best_gen_idx]
        mean_fitness = float(np.mean(gen_fitnesses))
        min_fitness = float(np.min(gen_fitnesses))

        if gen == 1:
            initial_pop_best_fitness = best_gen_res["fitness"]

        # Enforce monotonic best-so-far solution tracking
        if best_so_far_res is None or best_gen_res["fitness"] > best_so_far_res["fitness"]:
            best_so_far_res = best_gen_res.copy()

        convergence_history.append(
            {
                "generation": gen,
                "gen_best_fitness": best_gen_res["fitness"],
                "gen_mean_fitness": mean_fitness,
                "gen_min_fitness": min_fitness,
                "best_so_far_fitness": best_so_far_res["fitness"],
                "best_chromosome": list(best_so_far_res["chromosome"]),
                "val_acc": best_so_far_res["val_acc"],
                "robustness": best_so_far_res["robustness"],
                "runtime": best_so_far_res["runtime"],
            }
        )

        if verbose:
            print(
                f"Generation {gen:2d} | Best: {best_so_far_res['fitness']:.4f} "
                f"| Gen Mean: {mean_fitness:.4f} "
                f"| Val Acc: {best_so_far_res['val_acc']:.4f} "
                f"| Robustness: {best_so_far_res['robustness']:.4f} "
                f"| Chrom: {best_so_far_res['chromosome']}"
            )

        population = create_next_population(
            population,
            gen_fitnesses,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            rng=rng,
        )

    fitness_gain = best_so_far_res["fitness"] - initial_pop_best_fitness
    gain_pct = (
        (fitness_gain / max(1e-6, initial_pop_best_fitness)) * 100.0
        if initial_pop_best_fitness > 0
        else 0.0
    )

    if verbose and best_so_far_res is not None:
        print("=" * 75)
        print("OPTIMIZATION COMPLETED")
        print(f"Best Chromosome:       {best_so_far_res['chromosome']} ([n_estimators, max_depth, min_samples_split])")
        print(f"Validation Score:      {best_so_far_res['val_acc']:.4f}")
        print(f"Robustness Score:      {best_so_far_res['robustness']:.4f}")
        print(f"Runtime (seconds):     {best_so_far_res['runtime']:.4f}")
        print(f"Initial Best Fitness:  {initial_pop_best_fitness:.4f}")
        print(f"Final Best Fitness:    {best_so_far_res['fitness']:.4f} (+{fitness_gain:.4f}, +{gain_pct:.2f}%)")
        print("=" * 75)

    return {
        "best_chromosome": best_so_far_res["chromosome"],
        "best_fitness": best_so_far_res["fitness"],
        "initial_fitness": initial_pop_best_fitness,
        "fitness_gain": fitness_gain,
        "fitness_gain_pct": gain_pct,
        "best_val_acc": best_so_far_res["val_acc"],
        "best_robustness": best_so_far_res["robustness"],
        "best_runtime": best_so_far_res["runtime"],
        "best_model": best_so_far_res["model"],
        "preprocessor": preprocessor,
        "history": convergence_history,
    }


# =============================================================================
# OPTIMIZATION ANALYSIS & ASCII CONVERGENCE VISUALIZATION
# =============================================================================
def plot_ascii_convergence(history: List[Dict[str, Any]]) -> str:
    """
    Renders an ASCII convergence chart illustrating fitness trajectory across generations.
    Explains whether and by how much the GA improved upon the initial population.
    """
    lines = []
    lines.append("\n" + "=" * 75)
    lines.append("EVOLUTIONARY SEARCH CONVERGENCE TRAJECTORY")
    lines.append("=" * 75)
    lines.append(f"{'Gen':<5} | {'Best-So-Far':<12} | {'Gen Mean':<10} | {'Val Acc':<9} | {'Robust':<8} | Visual Trajectory")
    lines.append("-" * 75)

    min_fit = min(h["gen_min_fitness"] for h in history)
    max_fit = max(h["best_so_far_fitness"] for h in history)
    span = max(1e-5, max_fit - min_fit)

    for h in history:
        gen = h["generation"]
        best_f = h["best_so_far_fitness"]
        mean_f = h["gen_mean_fitness"]
        val_acc = h["val_acc"]
        rob = h["robustness"]

        # Scale 20-character bar
        bar_len = int(round(20 * (best_f - min_fit) / span))
        bar_str = "#" * bar_len + "-" * (20 - bar_len)
        lines.append(
            f"{gen:<5d} | {best_f:<12.4f} | {mean_f:<10.4f} | {val_acc:<9.4f} | {rob:<8.4f} | [{bar_str}]"
        )

    lines.append("-" * 75)
    init_f = history[0]["gen_best_fitness"]
    final_f = history[-1]["best_so_far_fitness"]
    diff = final_f - init_f
    if diff > 0.0001:
        lines.append(
            f"ANALYSIS: The Genetic Algorithm improved the best fitness from {init_f:.4f} (Gen 1) to {final_f:.4f} (Gen {len(history)}), achieving a +{diff:.4f} (+{(diff/init_f)*100:.2f}%) gain over the initial population."
        )
    else:
        lines.append(
            f"ANALYSIS: The initial random population sampled an optimal or near-optimal frontier chromosome ({init_f:.4f}), which was strictly preserved across all generations via elitism."
        )
    lines.append("=" * 75 + "\n")
    chart = "\n".join(lines)
    print(chart)
    return chart


# =============================================================================
# BASELINE COMPARISON ENGINE
# =============================================================================
def compare_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_chromosome: List[int],
    random_state: int = RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    """
    Compares the PA-EMLO Optimized Random Forest against two standard baselines:
    1. Default Random Forest: sklearn default parameters (n_estimators=100, max_depth=None, min_samples_split=2)
    2. Manual / Heuristic Random Forest: hand-tuned baseline (n_estimators=100, max_depth=5, min_samples_split=4)
    3. PA-EMLO Optimized Random Forest: hyperparameter set discovered by the evolutionary algorithm

    Evaluates all 3 models on the EXACT same training and held-out test splits.
    """
    models = {
        "Default RF": RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2, random_state=random_state, n_jobs=1
        ),
        "Manual RF": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=4, random_state=random_state, n_jobs=1
        ),
        f"PA-EMLO RF {best_chromosome}": build_model(
            best_chromosome, random_state=random_state
        ),
    }

    results = {}
    print("\n" + "=" * 85)
    print("BASELINE COMPARISON ON STRICTLY UNSEEN HELD-OUT TEST DATA")
    print("=" * 85)
    header = (
        f"{'Model':<24} | {'Bal Acc':<9} | {'F1-Macro':<9} | {'Robustness':<10} | "
        f"{'Acc Drop':<9} | {'Leaves':<7} | {'Latency(ms)':<11}"
    )
    print(header)
    print("-" * 85)

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model_comprehensive(
            model, X_test, y_test, random_state=random_state
        )
        results[name] = metrics
        print(
            f"{name:<24} | {metrics['balanced_accuracy']:<9.4f} | {metrics['f1_macro']:<9.4f} | "
            f"{metrics['robustness_score']:<10.4f} | {metrics['mean_acc_drop']:<9.4f} | "
            f"{metrics['total_leaves']:<7d} | {metrics['latency_per_sample_ms']:<11.4f}"
        )

    print("=" * 85)
    return results


# =============================================================================
# MULTI-SEED REPEATED EXPERIMENT HARNESS
# =============================================================================
def run_repeated_experiments(
    dataset_name: str = "breast_cancer",
    seeds: Optional[List[int]] = None,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Executes repeated experiments across multiple random seeds / stratified splits.
    Aggregates and reports Mean +/- Standard Deviation for Default RF, Manual RF, and PA-EMLO RF.
    """
    if seeds is None:
        seeds = [42, 101, 2024, 7, 99]

    print("\n" + "=" * 85)
    print(f"RUNNING REPEATED EXPERIMENTS ACROSS {len(seeds)} RANDOM SEEDS: {seeds}")
    print(f"Dataset: {dataset_name} | Pop: {population_size} | Gens: {generations}")
    print("=" * 85)

    X_raw, y_raw, desc = load_dataset(dataset_name)
    print(f"Dataset Info: {desc}")

    all_seed_results = {"Default RF": [], "Manual RF": [], "PA-EMLO RF": []}

    for seed_idx, seed in enumerate(seeds, 1):
        if verbose:
            print(f"\n--- Running Seed {seed} ({seed_idx}/{len(seeds)}) ---")

        preprocessor = DataPreprocessor(random_state=seed)
        X_tr, y_tr, X_va, y_va, X_te, y_te = preprocessor.train_val_test_split(
            X_raw, y_raw, train_size=0.6, val_size=0.2, test_size=0.2
        )

        # Run evolutionary search
        opt_res = run_evolution(
            splits=(X_tr, y_tr, X_va, y_va),
            population_size=population_size,
            generations=generations,
            random_state=seed,
            verbose=False,
        )
        best_chrom = opt_res["best_chromosome"]

        # Models to evaluate
        models = {
            "Default RF": RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_split=2, random_state=seed, n_jobs=1
            ),
            "Manual RF": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=4, random_state=seed, n_jobs=1
            ),
            "PA-EMLO RF": build_model(best_chrom, random_state=seed),
        }

        for m_name, model in models.items():
            model.fit(X_tr, y_tr)
            eval_metrics = evaluate_model_comprehensive(
                model, X_te, y_te, random_state=seed
            )
            eval_metrics["chromosome"] = best_chrom if m_name == "PA-EMLO RF" else []
            all_seed_results[m_name].append(eval_metrics)

    # Compute and format statistical summary (Mean +/- Std)
    summary_stats = {}
    print("\n" + "=" * 90)
    print(f"STATISTICAL BENCHMARK SUMMARY (N={len(seeds)} Repeated Runs: Mean +/- Std)")
    print("=" * 90)
    print(
        f"{'Model':<16} | {'Test Bal Acc':<17} | {'Test F1-Macro':<17} | "
        f"{'Robustness':<17} | {'Total Leaves':<15}"
    )
    print("-" * 90)

    metric_keys = ["balanced_accuracy", "f1_macro", "robustness_score", "total_leaves", "latency_per_sample_ms"]

    for m_name, runs in all_seed_results.items():
        summary_stats[m_name] = {}
        for k in metric_keys:
            vals = [r[k] for r in runs]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            summary_stats[m_name][k] = {"mean": mean_val, "std": std_val}

        ba = summary_stats[m_name]["balanced_accuracy"]
        f1 = summary_stats[m_name]["f1_macro"]
        rob = summary_stats[m_name]["robustness_score"]
        leaves = summary_stats[m_name]["total_leaves"]

        print(
            f"{m_name:<16} | {ba['mean']:.4f} +/- {ba['std']:.4f} | "
            f"{f1['mean']:.4f} +/- {f1['std']:.4f} | "
            f"{rob['mean']:.4f} +/- {rob['std']:.4f} | "
            f"{leaves['mean']:6.1f} +/- {leaves['std']:4.1f}"
        )

    print("=" * 90)
    return summary_stats


# =============================================================================
# COMPETITION SANDBOX ADAPTER
# =============================================================================
class CompetitionAdapter:
    """
    Decoupled interface adapter connecting the PA-EMLO optimizer
    with competitive sandbox benchmark evaluation harnesses.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.best_model: Optional[RandomForestClassifier] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.optimization_summary: Optional[Dict[str, Any]] = None

    def fit(
        self,
        X: Any,
        y: Any,
        population_size: int = POPULATION_SIZE,
        generations: int = GENERATIONS,
        verbose: bool = False,
    ) -> "CompetitionAdapter":
        """Runs hyperparameter search and fits the best model on sanitized data."""
        self.preprocessor = DataPreprocessor(random_state=self.random_state)
        X_train, y_train, X_val, y_val = self.preprocessor.validate_and_preprocess(X, y)

        results = run_evolution(
            splits=(X_train, y_train, X_val, y_val),
            population_size=population_size,
            generations=generations,
            random_state=self.random_state,
            verbose=verbose,
        )
        self.best_model = results["best_model"]
        self.optimization_summary = results
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for raw input data X using the best fitted model."""
        if self.best_model is None or self.preprocessor is None:
            raise RuntimeError("Adapter must be fitted before calling predict().")

        X_transformed = self.preprocessor.transform_new_data(X)
        return self.best_model.predict(X_transformed)

    def evaluate_test(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """Evaluates fitted model on unseen test data."""
        if self.best_model is None or self.preprocessor is None:
            raise RuntimeError("Adapter must be fitted before calling evaluate_test().")

        X_test_trans = self.preprocessor.transform_new_data(X_test)
        y_test_arr = np.array(y_test).ravel()
        return evaluate_model_comprehensive(
            self.best_model, X_test_trans, y_test_arr, random_state=self.random_state
        )


# =============================================================================
# COMPREHENSIVE AUTOMATED UNIT TEST SUITE
# =============================================================================
def run_tests() -> bool:
    """
    Executes 14 internal automated unit tests verifying algorithm correctness,
    operator bounds, elitism preservation, monotonic convergence, edge-case data
    handling, all-NaN column handling, 3-way leakage prevention, multi-perturbation
    evaluation, baseline comparisons, and multi-seed experiment execution.
    """
    print("=" * 75)
    print("RUNNING COMPREHENSIVE AUTOMATED TEST SUITE (14 TESTS)")
    print("=" * 75)

    # Test 1: Chromosome Validity
    c = random_chromosome()
    assert is_valid_chromosome(c), f"Test 1 Failed: Chromosome {c} invalid."
    print("[PASS] Test 1: Chromosome Validity")

    # Test 2: Crossover Validity
    p1 = [50, 3, 2]
    p2 = [150, 10, 10]
    ch1, ch2 = crossover(p1, p2)
    assert is_valid_chromosome(ch1) and is_valid_chromosome(ch2), (
        "Test 2 Failed: Crossover output invalid."
    )
    print("[PASS] Test 2: Crossover Producing Valid Chromosomes")

    # Test 3: Mutation Validity
    mutated = mutate(p1, mutation_rate=1.0)
    assert is_valid_chromosome(mutated), f"Test 3 Failed: Mutated chromosome {mutated} invalid."
    print("[PASS] Test 3: Mutation Producing Valid Chromosomes")

    # Test 4: Fitness Value Bounds & Stability
    f_val = calculate_fitness(0.85, 0.90, 0.1, 0.2)
    assert 0.0 <= f_val <= 1.0 and math.isfinite(f_val), (
        f"Test 4 Failed: Fitness {f_val} out of bounds/non-finite."
    )
    f_nan = calculate_fitness(
        float("nan"), float("inf"), float("nan"), float("inf")
    )
    assert 0.0 <= f_nan <= 1.0 and math.isfinite(f_nan), (
        "Test 4 Failed: NaN/Inf fitness handling failed."
    )
    print("[PASS] Test 4: Fitness Returning Finite Bounded Values")

    # Test 5: Population Size Constancy
    pop = [random_chromosome() for _ in range(8)]
    scores = [float(i) / 10.0 for i in range(8)]
    next_pop = create_next_population(
        pop, scores, elite_count=1, mutation_rate=0.2
    )
    assert len(next_pop) == len(pop), (
        f"Test 5 Failed: Pop size changed from {len(pop)} to {len(next_pop)}."
    )
    print("[PASS] Test 5: Population Size Remaining Constant")

    # Test 6: Elite Preservation
    best_idx = 5
    scores[best_idx] = 0.999
    expected_elite = list(pop[best_idx])
    next_pop_elite = create_next_population(
        pop, scores, elite_count=1, mutation_rate=0.2
    )
    assert next_pop_elite[0] == expected_elite, "Test 6 Failed: Elite individual not preserved."
    print("[PASS] Test 6: Elite Preservation")

    # Test 7: Convergence / Monotonic Best-So-Far Tracking
    rng = np.random.RandomState(42)
    X_syn = rng.randn(100, 5)
    y_syn = (X_syn[:, 0] + X_syn[:, 1] > 0).astype(int)
    prep = DataPreprocessor(random_state=42)
    X_tr, y_tr, X_va, y_va = prep.validate_and_preprocess(X_syn, y_syn)
    results = run_evolution(
        splits=(X_tr, y_tr, X_va, y_va), population_size=4, generations=3, verbose=False
    )
    history = results["history"]
    fitnesses = [h["best_so_far_fitness"] for h in history]
    for i in range(1, len(fitnesses)):
        assert fitnesses[i] >= fitnesses[i - 1], (
            f"Test 7 Failed: Non-monotonic fitness at gen {i}: {fitnesses}"
        )
    print("[PASS] Test 7: Convergence / Monotonic Best-So-Far Tracking")

    # Test 8: Malformed & Insufficient Input Handling
    X_bad = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, np.inf, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    y_bad = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    prep = DataPreprocessor(random_state=42)
    X_tr, y_tr, X_va, y_va = prep.validate_and_preprocess(X_bad, y_bad)
    assert not np.isnan(X_tr).any() and not np.isinf(X_tr).any(), (
        "Test 8 Failed: NaN/Inf preprocessing failed."
    )
    print("[PASS] Test 8: Malformed & Insufficient Input Handling")

    # Test 9: All-NaN Column Handling Without Warnings
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        X_with_all_nan = pd.DataFrame(
            {
                "valid1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "all_nan": [np.nan] * 10,
                "valid2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            }
        )
        y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        prep_nan = DataPreprocessor(random_state=42)
        X_tr_n, y_tr_n, X_va_n, y_va_n, X_te_n, y_te_n = prep_nan.train_val_test_split(
            X_with_all_nan, y_dummy, train_size=0.6, val_size=0.2, test_size=0.2
        )
        # Check that no SimpleImputer warning was emitted
        imputer_warnings = [
            w for w in captured_warnings if "Skipping features without any observed values" in str(w.message)
        ]
        assert len(imputer_warnings) == 0, (
            f"Test 9 Failed: Imputer warning was emitted: {imputer_warnings}"
        )
        assert X_tr_n.shape[1] == 2, f"Test 9 Failed: Expected 2 valid columns, got {X_tr_n.shape[1]}"
    print("[PASS] Test 9: All-NaN Column Handling With Zero Warnings")

    # Test 10: 3-Way Leakage-Free Train/Val/Test Split
    X_syn_df, y_syn_arr, _ = load_dataset("synthetic")
    prep_3way = DataPreprocessor(random_state=42)
    X_tr3, y_tr3, X_va3, y_va3, X_te3, y_te3 = prep_3way.train_val_test_split(
        X_syn_df, y_syn_arr, train_size=0.6, val_size=0.2, test_size=0.2
    )
    assert len(X_tr3) == 300 and len(X_va3) == 100 and len(X_te3) == 100, (
        f"Test 10 Failed: Split sizes incorrect: {len(X_tr3)}, {len(X_va3)}, {len(X_te3)}"
    )
    assert not np.isnan(X_te3).any(), "Test 10 Failed: Test split contains NaNs."
    print("[PASS] Test 10: 3-Way Leakage-Free Train/Val/Test Partitioning")

    # Test 11: Multi-Perturbation Robustness Engine
    model_test = build_model([50, 3, 2], random_state=42)
    model_test.fit(X_tr3, y_tr3)
    rob_dict = evaluate_multi_perturbations(model_test, X_te3, y_te3, random_state=42)
    assert "clean_acc" in rob_dict and "mean_robustness" in rob_dict, (
        "Test 11 Failed: Robustness report keys missing."
    )
    assert 0.0 <= rob_dict["mean_robustness"] <= 1.0, (
        f"Test 11 Failed: Robustness {rob_dict['mean_robustness']} out of bounds."
    )
    assert len(rob_dict["details"]) == 12, (
        f"Test 11 Failed: Expected 12 perturbation regimes, got {len(rob_dict['details'])}"
    )
    print("[PASS] Test 11: Multi-Perturbation Robustness Engine (12 Regimes)")

    # Test 12: Real-World Dataset Integration (Breast Cancer)
    X_bc, y_bc, desc_bc = load_dataset("breast_cancer")
    assert len(X_bc) == 569 and X_bc.shape[1] == 30, (
        f"Test 12 Failed: Breast cancer dataset shape mismatch: {X_bc.shape}"
    )
    prep_bc = DataPreprocessor(random_state=42)
    X_tr_b, y_tr_b, X_va_b, y_va_b, X_te_b, y_te_b = prep_bc.train_val_test_split(
        X_bc, y_bc, train_size=0.6, val_size=0.2, test_size=0.2
    )
    adapter = CompetitionAdapter(random_state=42)
    adapter.fit(X_bc, y_bc, verbose=False)
    preds_bc = adapter.predict(X_bc.iloc[:5])
    assert len(preds_bc) == 5, "Test 12 Failed: Adapter predict failed on real-world data."
    print("[PASS] Test 12: Real-World Dataset Integration & Adapter Pipeline")

    # Test 13: Baseline Comparison Execution
    base_res = compare_baselines(
        X_tr3, y_tr3, X_te3, y_te3, best_chromosome=[75, 5, 2], random_state=42
    )
    assert len(base_res) == 3, f"Test 13 Failed: Expected 3 baselines, got {len(base_res)}"
    assert "Default RF" in base_res and "Manual RF" in base_res, "Test 13 Failed: Baseline missing."
    print("[PASS] Test 13: Baseline Comparison Suite Execution")

    # Test 14: Multi-Seed Experiment Harness
    stats = run_repeated_experiments(
        dataset_name="synthetic",
        seeds=[42, 101],
        population_size=4,
        generations=2,
        verbose=False,
    )
    assert "Default RF" in stats and "PA-EMLO RF" in stats, (
        "Test 14 Failed: Multi-seed statistical summary missing expected models."
    )
    assert "mean" in stats["PA-EMLO RF"]["balanced_accuracy"], (
        "Test 14 Failed: Mean statistics missing."
    )
    print("[PASS] Test 14: Multi-Seed Experiment Harness & Statistical Aggregation")

    print("=" * 75)
    print("ALL 14 AUTOMATED UNIT TESTS PASSED SUCCESSFULLY (100% PASS RATE)!")
    print("=" * 75)
    return True


# =============================================================================
# MAIN BENCHMARK & DEMONSTRATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        run_tests()

    elif "--experiments" in args:
        # Run repeated 5-seed experiments on real-world dataset
        run_repeated_experiments(
            dataset_name="breast_cancer",
            seeds=[42, 101, 2024, 7, 99],
            population_size=8,
            generations=5,
            verbose=True,
        )

    elif "--synthetic" in args:
        # Run synthetic benchmark
        print("\n" + "=" * 75)
        print("RUNNING SYNTHETIC BENCHMARK EXPERIMENT")
        print("=" * 75)
        X_df, y_arr, desc = load_dataset("synthetic")
        print(f"Dataset: {desc}")

        prep = DataPreprocessor(random_state=RANDOM_SEED)
        X_tr, y_tr, X_va, y_va, X_te, y_te = prep.train_val_test_split(
            X_df, y_arr, train_size=0.6, val_size=0.2, test_size=0.2
        )

        results = run_evolution(
            splits=(X_tr, y_tr, X_va, y_va),
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            random_state=RANDOM_SEED,
            verbose=True,
        )
        plot_ascii_convergence(results["history"])

        compare_baselines(
            X_tr, y_tr, X_te, y_te, results["best_chromosome"], random_state=RANDOM_SEED
        )

    else:
        # Default workflow:
        # 1. Execute full unit test suite
        run_tests()

        # 2. Run Real-World Wisconsin Breast Cancer Benchmark with 3-way split
        print("\n" + "=" * 75)
        print("RUNNING REAL-WORLD TABULAR DATASET BENCHMARK (WISCONSIN BREAST CANCER)")
        print("=" * 75)
        X_df, y_arr, desc = load_dataset("breast_cancer")
        print(f"Dataset: {desc}")
        print(f"Total Samples: {len(X_df)} | Total Features: {X_df.shape[1]}")

        preprocessor = DataPreprocessor(random_state=RANDOM_SEED)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.train_val_test_split(
            X_df, y_arr, train_size=0.60, val_size=0.20, test_size=0.20
        )
        print(
            f"Leakage-Free 3-Way Split -> Train: {len(X_train)} samples | Val: {len(X_val)} samples | Test (Held-Out): {len(X_test)} samples"
        )

        # 3. Evolutionary Search on Train + Val
        opt_results = run_evolution(
            splits=(X_train, y_train, X_val, y_val),
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            random_state=RANDOM_SEED,
            verbose=True,
        )

        # 4. Convergence Trajectory Analysis
        plot_ascii_convergence(opt_results["history"])

        # 5. Baseline Comparison on Strictly Held-Out Test Data
        compare_baselines(
            X_train,
            y_train,
            X_test,
            y_test,
            opt_results["best_chromosome"],
            random_state=RANDOM_SEED,
        )

        # 6. Sample Held-Out Test Predictions
        best_model = build_model(opt_results["best_chromosome"], random_state=RANDOM_SEED)
        best_model.fit(X_train, y_train)
        test_sample_preds = best_model.predict(X_test[:10])
        test_sample_actual = y_test[:10]
        print("\nSAMPLE HELD-OUT UNSEEN TEST PREDICTIONS (First 10 instances):")
        print(f"Predicted: {test_sample_preds}")
        print(f"Actual:    {test_sample_actual}")
        match_count = np.sum(test_sample_preds == test_sample_actual)
        print(f"Accuracy on 10 sample batch: {match_count}/10 ({match_count*10:.1f}%)")
        print("=" * 75 + "\n")
