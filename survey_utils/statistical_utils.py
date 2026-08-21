"""Small, reusable helpers for the notebook's statistical test engines.

Test selection, thresholds, and analytical orchestration deliberately remain
visible in the notebook. These functions only implement generic diagnostics,
effect-size labels, and result wording.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def check_chi_square_assumptions(
    expected: np.ndarray,
    minimum_expected: float = 5,
    maximum_lt_minimum_percent: float = 20,
    minimum_expected_cell: float = 1,
) -> dict[str, float | int | bool]:
    """Evaluate Pearson chi-square expected-frequency assumptions."""
    cells_lt_minimum = int((expected < minimum_expected).sum())
    percent_lt_minimum = cells_lt_minimum / expected.size * 100
    any_lt_minimum_cell = bool((expected < minimum_expected_cell).any())
    assumption_ok = (
        not any_lt_minimum_cell
        and percent_lt_minimum <= maximum_lt_minimum_percent
    )
    return {
        "assumption_ok": assumption_ok,
        "cells_expected_lt_5": cells_lt_minimum,
        "percent_expected_lt_5": round(float(percent_lt_minimum), 1),
        "minimum_expected": round(float(expected.min()), 2),
        "any_expected_lt_1": any_lt_minimum_cell,
    }


def assess_group_normality(
    groups: Mapping[object, Sequence[float]],
    alpha: float,
) -> tuple[bool, dict[object, dict[str, float | bool]]]:
    """Assess Shapiro-Wilk normality within every supplied comparison group."""
    normality: dict[object, dict[str, float | bool]] = {}
    for name, values in groups.items():
        values_array = np.asarray(values, dtype=float)
        if len(values_array) < 3 or np.nanstd(values_array) == 0:
            p_value = np.nan
        else:
            _, p_value = stats.shapiro(values[:5000])
        normality[name] = {
            "p_value": p_value,
            "normal": bool(pd.notna(p_value) and p_value > alpha),
        }
    return all(result["normal"] for result in normality.values()), normality


def interpret_cramers_v(effect_size: float) -> str | float:
    """Return a conventional qualitative interpretation of Phi/Cramer's V."""
    if pd.isna(effect_size):
        return np.nan
    if effect_size < 0.10:
        return "Negligible"
    if effect_size < 0.30:
        return "Small"
    if effect_size < 0.50:
        return "Medium"
    return "Large"


def interpret_cohens_d(effect_size: float) -> str | float:
    """Return a conventional qualitative interpretation of Cohen's d."""
    effect_size = abs(effect_size)
    if pd.isna(effect_size):
        return np.nan
    if effect_size < 0.20:
        return "Negligible"
    if effect_size < 0.50:
        return "Small"
    if effect_size < 0.80:
        return "Medium"
    return "Large"


def interpret_eta_squared(effect_size: float) -> str | float:
    """Return a conventional qualitative interpretation of eta-squared."""
    if pd.isna(effect_size):
        return np.nan
    if effect_size < 0.01:
        return "Negligible"
    if effect_size < 0.06:
        return "Small"
    if effect_size < 0.14:
        return "Medium"
    return "Large"


def interpret_epsilon_squared(effect_size: float) -> str | float:
    """Return a conventional qualitative interpretation of epsilon-squared."""
    if pd.isna(effect_size):
        return np.nan
    if effect_size < 0.01:
        return "Negligible"
    if effect_size < 0.08:
        return "Small"
    if effect_size < 0.26:
        return "Medium"
    return "Large"


def interpret_rank_biserial(effect_size: float) -> str | float:
    """Return a conventional qualitative interpretation of rank-biserial r."""
    effect_size = abs(effect_size)
    if pd.isna(effect_size):
        return np.nan
    if effect_size < 0.10:
        return "Negligible"
    if effect_size < 0.30:
        return "Small"
    if effect_size < 0.50:
        return "Medium"
    return "Large"


def build_interpretation(
    p_value: float,
    effect_interpretation: str | float,
    alpha: float,
) -> str | float:
    """Return standardized significance and effect-size interpretation text."""
    if pd.isna(p_value):
        return np.nan
    significance = (
        "Statistically significant"
        if p_value < alpha
        else "No statistically significant"
    )
    if pd.isna(effect_interpretation):
        return significance
    return f"{significance} (effect size: {effect_interpretation.lower()})"


def categorical_group_test(
    data_: pd.DataFrame,
    var: str,
    group_var: str,
    *,
    config: Mapping[str, float | int],
    return_details: bool = False,
    skip_self_comparison: bool = False,
):
    """
    Compare a categorical variable across reporting groups.
    The function automatically evaluates Chi-square assumptions,selects the appropriate statistical test, calculates effect size, 
    and returns a standardized result suitable for automated reporting.
    When requested, diagnostic tables used during the analysis are also returned for quality assurance."""
    
    # ------------------------------------------------------------------
    # Invalid comparisons
    # ------------------------------------------------------------------

    if skip_self_comparison:

        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_name": np.nan,
            "n": 0,
            "interpretation": "Self-comparison skipped",

            "dof": np.nan,
            "assumption_ok": np.nan,
            "minimum_expected": np.nan
        }

        return (result, None) if return_details else result
    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    sub = data_[[var, group_var]].dropna()
    group_counts = sub[group_var].value_counts()

    if (
        len(sub) < config["min_sample_size"]
        or group_counts.empty
        or group_counts.min() < config["min_group_size"]
        or sub[var].nunique() < 2
        or sub[group_var].nunique() < 2
    ):

        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_name": np.nan,
            "n": len(sub),
            "interpretation": "Not tested — insufficient base",
            "status": "suppressed",
            "suppression_reason": (
                f"Requires total n >= {config['min_sample_size']} and "
                f"each group n >= {config['min_group_size']}."
            ),

            "dof": np.nan,
            "assumption_ok": np.nan,
            "minimum_expected": np.nan
        }

        return (result, None) if return_details else result

    # ------------------------------------------------------------------
    # Contingency table
    # ------------------------------------------------------------------

    observed = pd.crosstab(sub[var], sub[group_var])

    chi2, p_chi, dof, expected = stats.chi2_contingency(observed,correction=False)

    expected = pd.DataFrame(expected, index=observed.index, columns=observed.columns)

    residuals = (observed - expected) / np.sqrt(expected)

    # ------------------------------------------------------------------
    # Effect size
    # ------------------------------------------------------------------

    n = observed.to_numpy().sum()
    min_dim = min(observed.shape) - 1

    effect_size = (np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan)

    effect_name = ("Phi" if observed.shape == (2, 2) else "Cramer's V")

    effect_interpretation = interpret_cramers_v(effect_size)

    # ------------------------------------------------------------------
    # Assumption checks
    # ------------------------------------------------------------------

    assumptions = check_chi_square_assumptions(
        expected.to_numpy(),
        minimum_expected=config["chi_square_min_expected"],
        maximum_lt_minimum_percent=config["chi_square_max_lt5_percent"],
        minimum_expected_cell=config["chi_square_min_expected_cell"],
    )

    # ------------------------------------------------------------------
    # Statistical test selection
    # ------------------------------------------------------------------

    odds_ratio = np.nan

    if assumptions["assumption_ok"]:

        test = "Pearson Chi-square"
        statistic = chi2
        p_value = p_chi
        dof_out = dof

    elif observed.shape == (2, 2):

        odds_ratio, p_value = stats.fisher_exact(observed)

        test = "Fisher's Exact"
        statistic = np.nan
        dof_out = np.nan

    else:
        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": round(float(effect_size), 3),
            "effect_name": effect_name,
            "n": int(n),
            "interpretation": "Not tested — sparse contingency table",
            "status": "suppressed",
            "suppression_reason": (
                "Pearson assumptions failed and an exact test is not "
                "implemented for tables larger than 2×2."
            ),
            "dof": int(dof),
            "assumption_ok": False,
            "minimum_expected": assumptions["minimum_expected"],
            "cells_expected_lt_5": assumptions["cells_expected_lt_5"],
            "percent_expected_lt_5": assumptions["percent_expected_lt_5"],
            "any_expected_lt_1": assumptions["any_expected_lt_1"],
            "effect_interpretation": effect_interpretation,
            "odds_ratio": np.nan,
        }
        details = {
            "observed": observed,
            "expected": expected,
            "pearson_residuals": residuals,
        }
        return (result, details) if return_details else result

    # ------------------------------------------------------------------
    # Standardized result
    # ------------------------------------------------------------------

    result = {

        # Common fields
        "variable": var,
        "breakdown": group_var,
        "test": test,

        "statistic": (
            round(float(statistic), 3)
            if pd.notna(statistic)
            else np.nan
        ),

        "p_value": round(float(p_value), 4),

        "effect_size": (
            round(float(effect_size), 3)
            if pd.notna(effect_size)
            else np.nan
        ),

        "effect_name": effect_name,

        "n": int(n),

        "interpretation": build_interpretation(p_value, effect_interpretation, config["alpha"]),
        "status": "completed",
        "suppression_reason": None,

        # Test-specific fields
        "dof": (
            int(dof_out)
            if pd.notna(dof_out)
            else np.nan
        ),

        "assumption_ok": assumptions["assumption_ok"],
        "minimum_expected": assumptions["minimum_expected"],

        "cells_expected_lt_5":
            assumptions["cells_expected_lt_5"],

        "percent_expected_lt_5":
            assumptions["percent_expected_lt_5"],

        "any_expected_lt_1":
            assumptions["any_expected_lt_1"],

        "effect_interpretation":
            effect_interpretation,

        "odds_ratio": (round(float(odds_ratio), 3) if pd.notna(odds_ratio) else np.nan)
    }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    details = {

        "observed": observed,

        "expected": expected,

        "pearson_residuals": residuals

    }

    return (result, details) if return_details else result


def survey_categorical_group_test(
    data_: pd.DataFrame,
    var: str,
    group_var: str,
    *,
    config: Mapping[str, float | int],
    weight_col: str,
    strata_cols: Sequence[str],
    fpc_col: str | None = None,
    psu_col: str | None = None,
    return_details: bool = False,
    skip_self_comparison: bool = False,
):
    """Run a design-adjusted categorical comparison for survey data.

    The function uses sampling weights, strata, and optional finite-population
    and PSU fields supplied by the caller. It reports the design-adjusted
    Rao–Scott F approximation and a weighted Cramer's V/Phi as a descriptive
    effect-size companion.
    """
    required = [var, group_var, weight_col, *strata_cols]
    if fpc_col:
        required.append(fpc_col)
    if psu_col:
        required.append(psu_col)
    missing = [column for column in required if column not in data_.columns]
    if missing:
        raise KeyError(
            "Survey-design comparison is missing required columns: "
            + ", ".join(missing)
        )

    if skip_self_comparison:
        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_name": np.nan,
            "n": 0,
            "interpretation": "Self-comparison skipped",
            "status": "suppressed",
            "suppression_reason": "The variable and breakdown represent the same construct.",
            "dof": np.nan,
            "denominator_dof": np.nan,
        }
        return (result, None) if return_details else result

    selected_columns = list(dict.fromkeys(required))
    sub = data_[selected_columns].copy()
    sub[weight_col] = pd.to_numeric(sub[weight_col], errors="coerce")
    valid = (
        sub[var].notna()
        & sub[group_var].notna()
        & sub[weight_col].notna()
        & sub[weight_col].gt(0)
    )
    valid &= sub[strata_cols].notna().all(axis=1)
    if fpc_col:
        sub[fpc_col] = pd.to_numeric(sub[fpc_col], errors="coerce")
        valid &= sub[fpc_col].notna() & sub[fpc_col].gt(0)
    if psu_col:
        valid &= sub[psu_col].notna()
    sub = sub.loc[valid].copy()

    group_counts = sub[group_var].value_counts()
    category_counts = sub[var].value_counts()
    min_category_size = int(config.get("min_category_size", 1))
    insufficient = (
        len(sub) < int(config["min_sample_size"])
        or group_counts.empty
        or group_counts.min() < int(config["min_group_size"])
        or category_counts.empty
        or category_counts.min() < min_category_size
        or sub[var].nunique() < 2
        or sub[group_var].nunique() < 2
    )
    if insufficient:
        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_name": np.nan,
            "n": len(sub),
            "interpretation": "Not tested — insufficient base",
            "status": "suppressed",
            "suppression_reason": (
                f"Requires total n >= {config['min_sample_size']}, each comparison "
                f"group n >= {config['min_group_size']}, and each response category "
                f"n >= {min_category_size}."
            ),
            "dof": np.nan,
            "denominator_dof": np.nan,
        }
        return (result, None) if return_details else result

    stratum_field = "__survey_stratum"
    sub[stratum_field] = (
        sub[strata_cols].astype(str).agg(" | ".join, axis=1)
        if len(strata_cols) > 1
        else sub[strata_cols[0]].astype(str)
    )

    observed = pd.crosstab(sub[var], sub[group_var])
    weighted_observed = pd.pivot_table(
        sub,
        values=weight_col,
        index=var,
        columns=group_var,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    weighted_chi2, _, _, weighted_expected = stats.chi2_contingency(
        weighted_observed, correction=False
    )
    weighted_total = float(weighted_observed.to_numpy().sum())
    min_dim = min(weighted_observed.shape) - 1
    effect_size = (
        np.sqrt(weighted_chi2 / (weighted_total * min_dim))
        if weighted_total > 0 and min_dim > 0
        else np.nan
    )
    effect_name = (
        "Weighted Phi"
        if weighted_observed.shape == (2, 2)
        else "Weighted Cramer's V"
    )
    effect_interpretation = interpret_cramers_v(effect_size)

    try:
        import polars as pl
        import svy
    except ImportError as exc:
        raise ImportError(
            "Survey-design inference requires the optional 'svy' and 'polars' "
            "packages. Install them explicitly or choose INFERENCE_MODE='none' "
            "or a methodologically justified 'standard' plan."
        ) from exc

    design_columns = [var, group_var, weight_col, stratum_field]
    if fpc_col:
        design_columns.append(fpc_col)
    if psu_col:
        design_columns.append(psu_col)
    design_frame = sub[design_columns].reset_index(drop=True)
    for column in [var, group_var, stratum_field]:
        design_frame[column] = design_frame[column].astype(str)

    design = svy.Design(
        stratum=stratum_field,
        wgt=weight_col,
        pop_size=fpc_col,
        psu=psu_col,
    )
    sample = svy.Sample(pl.from_pandas(design_frame), design)
    table = sample.categorical.tabulate(rowvar=var, colvar=group_var)
    f_stat = table.stats.f
    chi_stat = table.stats.chisq
    p_value = float(f_stat.p_value)

    result = {
        "variable": var,
        "breakdown": group_var,
        "test": "Rao-Scott chi-square",
        "statistic": round(float(f_stat.value), 3),
        "p_value": round(p_value, 4),
        "effect_size": (
            round(float(effect_size), 3) if pd.notna(effect_size) else np.nan
        ),
        "effect_name": effect_name,
        "n": int(len(sub)),
        "interpretation": build_interpretation(
            p_value, effect_interpretation, float(config["alpha"])
        ),
        "status": "completed",
        "suppression_reason": None,
        "dof": round(float(f_stat.df_num), 3),
        "denominator_dof": round(float(f_stat.df_den), 3),
        "rao_scott_chi_square": round(float(chi_stat.value), 3),
        "assumption_ok": np.nan,
        "minimum_expected": round(float(np.min(weighted_expected)), 2),
        "effect_interpretation": effect_interpretation,
        "odds_ratio": np.nan,
    }
    details = {
        "observed": observed,
        "weighted_observed": weighted_observed,
        "weighted_expected": pd.DataFrame(
            weighted_expected,
            index=weighted_observed.index,
            columns=weighted_observed.columns,
        ),
    }
    return (result, details) if return_details else result


def continuous_group_test(
    data_: pd.DataFrame,
    var: str,
    group_var: str,
    *,
    config: Mapping[str, float | int],
    return_details: bool = False,
):
    """
    Compare a continuous variable across reporting groups.

    The function automatically evaluates normality assumptions,
    selects the appropriate statistical test, calculates effect size,
    and returns a standardized result suitable for automated reporting.

    When requested, diagnostic summaries used during the analysis are
    also returned for quality assurance.
    """

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    sub = data_[[var, group_var]].copy()

    sub[var] = pd.to_numeric(sub[var], errors="coerce")
    sub = sub.dropna()

    grouped = {
        name: group[var].values
        for name, group in sub.groupby(group_var)
    }
    group_counts = sub[group_var].value_counts()
    k_groups = len(grouped)

    if (
        len(sub) < config["min_sample_size"]
        or k_groups < 2
        or group_counts.min() < config["min_group_size"]
    ):

        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_name": np.nan,
            "n": len(sub),
            "interpretation": "Not tested — insufficient base",
            "status": "suppressed",
            "suppression_reason": (
                f"Requires total n >= {config['min_sample_size']}, at least two "
                f"groups, and n >= {config['min_group_size']} in every group."
            ),

            "k_groups": k_groups,
            "normality_assumption": np.nan
        }

        return (result, None) if return_details else result

    if all(np.nanstd(values) == 0 for values in grouped.values()):
        result = {
            "variable": var,
            "breakdown": group_var,
            "test": None,
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "effect_name": np.nan,
            "n": len(sub),
            "interpretation": "Not tested — no within-group variation",
            "status": "suppressed",
            "suppression_reason": "Every retained comparison group has zero variance.",
            "k_groups": k_groups,
            "normality_assumption": np.nan,
        }
        return (result, None) if return_details else result

    # ------------------------------------------------------------------
    # Normality assessment
    # ------------------------------------------------------------------

    normality_assumption, normality = assess_group_normality(grouped, config["alpha"])

    groups = list(grouped.values())

    # ------------------------------------------------------------------
    # Statistical test selection
    # ------------------------------------------------------------------

    statistic = np.nan
    p_value = np.nan
    effect_size = np.nan
    effect_name = np.nan
    effect_interpretation = np.nan

    if k_groups == 2:

        if normality_assumption:

            statistic, p_value = stats.ttest_ind(
                groups[0],
                groups[1],
                equal_var=False
            )

            n1, n2 = len(groups[0]), len(groups[1])
            pooled_variance = (
                (n1 - 1) * np.var(groups[0], ddof=1)
                + (n2 - 1) * np.var(groups[1], ddof=1)
            ) / (n1 + n2 - 2)
            pooled_sd = np.sqrt(pooled_variance)

            effect_size = (
                (np.mean(groups[0]) - np.mean(groups[1])) / pooled_sd
                if pooled_sd > 0
                else np.nan
            )

            effect_name = "Cohen's d"
            effect_interpretation = interpret_cohens_d(effect_size)
            test = "Welch t-test"

        else:

            statistic, p_value = stats.mannwhitneyu(
                groups[0],
                groups[1],
                alternative="two-sided"
            )

            n1 = len(groups[0])
            n2 = len(groups[1])

            effect_size = 1 - (2 * statistic) / (n1 * n2)

            effect_name = "Rank-biserial r"
            effect_interpretation = interpret_rank_biserial(effect_size)
            test = "Mann-Whitney U"

    else:

        if normality_assumption:

            try:

                from statsmodels.stats.oneway import anova_oneway

                res = anova_oneway(
                    groups,
                    use_var="unequal"
                )

                statistic = res.statistic
                p_value = res.pvalue

                test = "Welch ANOVA"

            except Exception:

                statistic, p_value = stats.f_oneway(*groups)

                test = "One-way ANOVA"

            all_values = np.concatenate(groups)

            grand_mean = all_values.mean()

            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            )

            ss_total = (
                (all_values - grand_mean) ** 2
            ).sum()

            effect_size = (
                ss_between / ss_total
                if ss_total > 0
                else np.nan
            )

            effect_name = "Eta-squared"
            effect_interpretation = interpret_eta_squared(effect_size)

        else:

            statistic, p_value = stats.kruskal(*groups)

            effect_size = (
                max(0.0, (statistic - k_groups + 1) / (len(sub) - k_groups))
                if (len(sub) - k_groups) > 0
                else np.nan
            )

            effect_name = "Epsilon-squared"
            effect_interpretation = interpret_epsilon_squared(effect_size)

            test = "Kruskal-Wallis"

    # ------------------------------------------------------------------
    # Standardized result
    # ------------------------------------------------------------------

    result = {

        # Common fields
        "variable": var,
        "breakdown": group_var,
        "test": test,

        "statistic": round(float(statistic), 3),
        "p_value": round(float(p_value), 4),

        "effect_size": (
            round(float(effect_size), 3)
            if pd.notna(effect_size)
            else np.nan
        ),

        "effect_name": effect_name,

        "n": int(len(sub)),

        "interpretation": build_interpretation(
            p_value,
            effect_interpretation,
            config["alpha"],
        ),
        "status": "completed",
        "suppression_reason": None,

        # Test-specific fields
        "k_groups": k_groups,
        "normality_assumption": normality_assumption,

        "effect_interpretation": effect_interpretation
    }

    # ------------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------------

    group_summary = []

    for name, values in grouped.items():

        group_summary.append({

            "Group": name,
            "N": len(values),
            "Mean": round(float(np.mean(values)), 2),
            "Median": round(float(np.median(values)), 2),
            "SD": round(float(np.std(values, ddof=1)), 2)

        })

    details = {

        "group_summary": pd.DataFrame(group_summary),

        "normality": pd.DataFrame(normality).T

    }

    return (result, details) if return_details else result
