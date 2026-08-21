"""Generic data-quality, descriptive, and indicator calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .kobo_metadata import (
    compute_relevant_mask,
    display_category,
    get_multi_binary,
    get_question_label,
    ordered_response_codes,
)


def resolve_identifier_column(
    data: pd.DataFrame,
    preferred_column: str | None = None,
    fallback_column: str | None = "_index",
) -> str | None:
    """Resolve a response reference using preferred, Kobo, then index fallback.

    A unique case-insensitive match supports Kobo exports that alter the case
    of a configured identifier field. `None` means callers should use the
    dataframe index.
    """
    if preferred_column:
        if preferred_column in data.columns:
            return preferred_column
        matches = [column for column in data.columns if column.lower() == preferred_column.lower()]
        if len(matches) == 1:
            return matches[0]
    if fallback_column and fallback_column in data.columns:
        return fallback_column
    return None


def record_identifiers(
    data: pd.DataFrame,
    mask: pd.Series,
    identifier_column: str | None,
) -> list[Any]:
    """Return record references for a boolean mask using the resolved identifier."""
    if identifier_column and identifier_column in data.columns:
        return data.loc[mask, identifier_column].tolist()
    return data.loc[mask].index.tolist()


def iqr_outlier_summary(series: pd.Series) -> pd.Series:
    """Return IQR screening statistics for one continuous variable."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.Series(
            {"N": 0, "Min": np.nan, "Max": np.nan, "Lower_Bound": np.nan,
             "Upper_Bound": np.nan, "N_Outliers": 0}
        )
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return pd.Series(
        {"N": len(values), "Min": values.min(), "Max": values.max(),
         "Lower_Bound": round(lower, 1), "Upper_Bound": round(upper, 1),
         "N_Outliers": int(((values < lower) | (values > upper)).sum())}
    )


def eda_weighting_label(weight_col: str | None = None) -> str:
    """Return an explicit label describing the EDA weighting basis."""
    return f"Weighted ({weight_col})" if weight_col else "Unweighted"


def eda_categorical_summary(
    data: pd.DataFrame, variable: str, label: str | None = None,
    section: str | None = None, weight_col: str | None = None,
) -> pd.DataFrame:
    """Create a compact categorical EDA summary."""
    if variable not in data.columns or data.empty:
        return pd.DataFrame()
    values = data[variable]
    counts = values.value_counts(dropna=False)
    if counts.empty:
        return pd.DataFrame()
    percent = counts / len(values) * 100
    return pd.DataFrame({
        "Variable": [variable], "Label": [label], "Section": [section],
        "Weighting": [eda_weighting_label(weight_col)],
        "N": [values.notna().sum()], "Missing": [values.isna().sum()],
        "Categories": [values.nunique(dropna=True)],
        "Largest category": [f"{counts.idxmax()} ({counts.max()}, {percent.loc[counts.idxmax()]:.1f}%)"],
        "Smallest category": [f"{counts.idxmin()} ({counts.min()}, {percent.loc[counts.idxmin()]:.1f}%)"],
    })


def eda_continuous_summary(data: pd.DataFrame, variable: str, weight_col: str | None = None) -> pd.DataFrame:
    """Summarize continuous-variable completeness."""
    if variable not in data.columns:
        return pd.DataFrame()
    return pd.DataFrame({"Weighting": [eda_weighting_label(weight_col)],
                         "N": [data[variable].notna().sum()],
                         "Missing": [data[variable].isna().sum()]})


def eda_multiple_summary(
    choice_maps: Mapping[str, Mapping[str, str]], list_name: str | None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Summarize available choices for a multiple-response question."""
    choices = choice_maps.get(list_name, {})
    if not choices:
        return pd.DataFrame()
    return pd.DataFrame({"Weighting": [eda_weighting_label(weight_col)],
                         "Number of choices": [len(choices)]})


def weighted_quantile(values: Sequence[float], quantiles: Sequence[float], sample_weight: Sequence[float]) -> np.ndarray:
    """Calculate weighted quantiles, returning NaNs when no valid data exist."""
    values_array = np.asarray(values, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights) & (weights >= 0)
    values_array, weights = values_array[valid], weights[valid]
    if not len(values_array) or weights.sum() == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values_array)
    values_array, weights = values_array[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    return np.interp(np.asarray(quantiles) * weights.sum(), cumulative, values_array)


def categorical_descriptives(
    data: pd.DataFrame, variable: str, list_name_by_var: Mapping[str, str | None],
    choice_maps: Mapping[str, Mapping[str, str]], weight_col: str | None = "Sampling_Weight",
    applicable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Calculate skip-aware categorical frequencies and percentages."""
    if variable not in data.columns:
        return pd.DataFrame()
    applicable = (
        applicable_mask.reindex(data.index, fill_value=False).astype(bool)
        if applicable_mask is not None
        else pd.Series(True, index=data.index)
    )
    values = data.loc[applicable, variable].dropna()
    if values.empty:
        return pd.DataFrame()
    order = ordered_response_codes(variable, values.unique(), list_name_by_var, choice_maps)
    frequency = values.value_counts().reindex(order, fill_value=0)
    output = pd.DataFrame({
        "Frequency": frequency,
        "Percent": (frequency / len(values) * 100).round(1),
    })
    if weight_col and weight_col in data.columns:
        weighted = (
            data.loc[values.index]
            .groupby(values)[weight_col]
            .sum()
            .reindex(order, fill_value=0)
        )
        output["Weighted_Percent"] = (weighted / weighted.sum() * 100).round(1)
    output.index = [display_category(value, variable, list_name_by_var, choice_maps) for value in output.index]
    output["Valid_N"] = len(values)
    output["Missing_N"] = int((applicable & data[variable].isna()).sum())
    output["Not_Applicable_N"] = int((~applicable).sum())
    return output


def continuous_descriptives(data: pd.DataFrame, variable: str, weight_col: str | None = None) -> pd.DataFrame:
    """Calculate standard continuous descriptive measures."""
    if variable not in data.columns:
        return pd.DataFrame()
    values = pd.to_numeric(data[variable], errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame()
    if weight_col and weight_col in data.columns:
        weights = pd.to_numeric(data.loc[values.index, weight_col], errors="coerce")
        valid = weights.notna() & (weights >= 0)
        values, weights = values.loc[valid], weights.loc[valid]
        q1, median, q3 = weighted_quantile(values, [0.25, 0.5, 0.75], weights)
        mean = np.average(values, weights=weights) if weights.sum() else np.nan
        sd = np.sqrt(np.average((values - mean) ** 2, weights=weights)) if weights.sum() else np.nan
    else:
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        mean, sd = values.mean(), values.std()
    return pd.DataFrame({"N": [len(values)], "Mean": [round(mean, 2)], "Median": [round(median, 2)],
                         "SD": [round(sd, 2)], "IQR": [round(q3 - q1, 2)],
                         "Min": [round(values.min(), 2)], "Max": [round(values.max(), 2)],
                         "Missing": [int(data[variable].isna().sum())]})


def multi_response_descriptives(
    data: pd.DataFrame, variable: str, list_name: str | None,
    choice_maps: Mapping[str, Mapping[str, str]], weight_col: str | None = None,
    applicable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Calculate skip-aware option-level multiple-response percentages."""
    options = choice_maps.get(list_name, {})
    if variable not in data.columns or not options:
        return pd.DataFrame()
    applicable = (
        applicable_mask.reindex(data.index, fill_value=False).astype(bool)
        if applicable_mask is not None
        else pd.Series(True, index=data.index)
    )
    valid = applicable & data[variable].notna()
    respondent_n = int(valid.sum())
    denominator_weight = (
        data.loc[valid, weight_col].sum()
        if weight_col and weight_col in data.columns
        else None
    )
    rows = []
    for code, label in options.items():
        selected = get_multi_binary(data, variable, code).astype(bool) & valid
        row = {"Option": label, "Frequency": int(selected.sum()),
               "Respondent_%": round(selected.sum() / respondent_n * 100, 1) if respondent_n else np.nan,
               "Valid_N": respondent_n,
               "Missing_N": int((applicable & data[variable].isna()).sum()),
               "Not_Applicable_N": int((~applicable).sum())}
        if weight_col and weight_col in data.columns:
            row["Weighted_Respondent_%"] = round(data.loc[selected, weight_col].sum() / denominator_weight * 100, 1) if denominator_weight else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_group_percentages(
    data: pd.DataFrame, variable: str, group_var: str, weight_col: str | None = "Sampling_Weight",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series] | None:
    """Calculate categorical group counts, row percentages, and overall percentages."""
    columns = [variable, group_var] + ([weight_col] if weight_col and weight_col in data.columns else [])
    if variable not in data.columns or group_var not in data.columns:
        return None
    subset = data[columns].dropna()
    if subset.empty:
        return None
    counts = pd.crosstab(subset[group_var], subset[variable])
    if weight_col and weight_col in subset.columns:
        weighted = subset.pivot_table(values=weight_col, index=group_var, columns=variable, aggfunc="sum", fill_value=0)
        report = weighted.div(weighted.sum(axis=1), axis=0).mul(100)
        overall = weighted.sum(axis=0).div(weighted.values.sum()).mul(100)
    else:
        report = counts.div(counts.sum(axis=1), axis=0).mul(100)
        overall = counts.sum(axis=0).div(counts.to_numpy().sum()).mul(100)
    return counts, report, overall


def calculate_indicator(
    data: pd.DataFrame, indicator_var: str, indicator_name: str, target: float,
    weight_col: str | None = "Sampling_Weight",) -> dict[str, Any] | None:
    """Calculate a binary or 0–1 composite indicator.

    Numeric scores support composite indicators such as the arithmetic mean of
    two component outcomes. Missing scores remain outside the denominator.
    """

    if indicator_var not in data.columns:
        return None
    if weight_col and weight_col not in data.columns:
        return None
    indicator = pd.to_numeric(data[indicator_var], errors="coerce")
    valid = indicator.notna() & indicator.between(0, 1)
    numerator = float(indicator.loc[valid].sum())
    denominator = int(valid.sum())
    unweighted_pct = (round(numerator / denominator * 100, 1) if denominator > 0 else np.nan)

    if weight_col:
        weights = pd.to_numeric(data[weight_col], errors="coerce")
        weighted_valid = valid & weights.notna() & (weights >= 0)
        weighted_denominator = data.loc[weighted_valid, weight_col,].sum()
        weighted_numerator = (
            indicator.loc[weighted_valid] * weights.loc[weighted_valid]
        ).sum()
        weighted_pct = (round(weighted_numerator / weighted_denominator * 100,1,) if weighted_denominator > 0 else np.nan)
    else:
        weighted_numerator = np.nan
        weighted_denominator = np.nan
        weighted_pct = np.nan

    return {
        "Indicator": indicator_name,
        "Target (%)": target,
        "Numerator / score sum": round(numerator, 2),
        "Denominator": denominator,
        "Missing / excluded": int((~valid).sum()),
        "Weighted numerator": round(float(weighted_numerator), 2) if pd.notna(weighted_numerator) else np.nan,
        "Weighted denominator": round(float(weighted_denominator), 2) if pd.notna(weighted_denominator) else np.nan,
        "Unweighted (%)": unweighted_pct,
        "Weighted (%)": weighted_pct,
        "Target Achieved": ("Yes" if pd.notna(weighted_pct) and weighted_pct >= target else "No"),}


QUESTION_ANALYSIS_COLUMNS = [
    "question_code", "question_label", "question_type", "response_code",
    "response_label", "analysis_level", "frequency", "unweighted_denominator",
    "weighted_frequency", "weighted_denominator", "unweighted_percentage",
    "weighted_percentage", "mean", "median", "sd", "missing_n", "not_applicable_n",
]


def _valid_weight_mask(data: pd.DataFrame, weight_col: str | None) -> tuple[pd.Series, pd.Series]:
    """Return usable sampling-weight values and their validity mask."""
    if not weight_col or weight_col not in data.columns:
        empty = pd.Series(False, index=data.index)
        return pd.Series(np.nan, index=data.index, dtype=float), empty
    weights = pd.to_numeric(data[weight_col], errors="coerce")
    return weights, weights.notna() & (weights >= 0)


def build_question_analysis_table(
    data: pd.DataFrame,
    qmeta: pd.DataFrame,
    *,
    indicator_labels: Mapping[str, str] | None = None,
    analysis_levels: Mapping[str, Sequence[str]] | None = None,
    breakdown_labels: Mapping[str, str] | None = None,
    choice_maps: Mapping[str, Mapping[str, str]] | None = None,
    weight_col: str | None = "Sampling_Weight",
    excluded_level_pairs: Mapping[str, Sequence[str]] | None = None,
    decimals: int = 1,
) -> pd.DataFrame:
    """Build static, question-based survey analysis results for Tableau.

    Every percentage uses the valid respondents to the same question within
    the configured analysis level as its denominator. This makes the result
    ready for static Tableau visuals: Tableau selects an `analysis_level` and
    displays the pre-calculated percentage rather than summing percentages.

    Args:
        data: Respondent-level analytical data limited to required variables.
        qmeta: Survey metadata containing `name`, `label`, `analysis_type`, and
            optionally `list_name` and `relevant_chain`.
        indicator_labels: Mapping of respondent-level indicator variables to
            analyst-facing labels.
        analysis_levels: Mapping of visible analysis-level names to the
            breakdown columns that define each level. `Overall` uses an empty
            sequence. Inactive breakdowns are exported as `All`.
        breakdown_labels: Optional output labels for breakdown columns.
        choice_maps: Kobo choice-list code-to-label mappings.
        weight_col: Optional sampling-weight column.
        excluded_level_pairs: Optional survey-specific mapping of variables to
            breakdown columns that should not be compared with that variable.
        decimals: Decimal places for percentages and continuous summaries.

    Returns:
        A long-format dataframe. `All` means a breakdown is inactive for that
        row; `Missing` is an observed missing breakdown value. `missing_n` counts
        unanswered applicable records, while `not_applicable_n` counts records
        excluded by XLSForm relevance/skip logic. Both are excluded from the
        question-level denominator.
    """
    required_metadata = {"name", "label", "analysis_type"}
    missing_metadata = required_metadata - set(qmeta.columns)
    if missing_metadata:
        raise ValueError(
            "qmeta is missing required columns: "
            f"{', '.join(sorted(missing_metadata))}."
        )

    choice_maps = choice_maps or {}
    indicator_labels = indicator_labels or {}
    breakdown_labels = breakdown_labels or {}
    excluded_level_pairs = excluded_level_pairs or {}
    if analysis_levels is None:
        analysis_levels = {"Overall": ()}
    configured_breakdowns = list(dict.fromkeys(
        column for columns in analysis_levels.values() for column in columns
    ))
    available_breakdowns = [column for column in configured_breakdowns if column in data.columns]
    output_breakdowns = [breakdown_labels.get(column, column) for column in available_breakdowns]
    list_name_by_var = (
        dict(zip(qmeta["name"], qmeta["list_name"]))
        if "list_name" in qmeta.columns else {}
    )

    working = data.copy()
    breakdown_group_columns: dict[str, str] = {}
    for index, column in enumerate(available_breakdowns):
        group_column = f"__aggregation_breakdown_{index}"
        breakdown_group_columns[column] = group_column
        display_values = working[column].where(working[column].notna(), "Missing")
        working[group_column] = display_values.map(
            lambda value: display_category(value, column, list_name_by_var, choice_maps)
        )

    metadata = qmeta.loc[
        qmeta["analysis_type"].isin(
            ["categorical_single", "continuous", "multiple_response"]
        ) & qmeta["name"].isin(working.columns)
    ].copy()
    questions = [
        (
            row["name"],
            row["analysis_type"],
            get_question_label(qmeta, row["name"]),
            row.get("list_name"),
            row.get("relevant_chain", ()),
        )
        for _, row in metadata.iterrows()
    ]
    questions.extend(
        (variable, "indicator", label, None, ())
        for variable, label in indicator_labels.items()
        if variable in working.columns
    )

    rows: list[dict[str, Any]] = []
    missing_multiple_choice_lists: set[str] = set()
    incomplete_multiple_response_data: set[str] = set()

    for level_name, configured_columns in analysis_levels.items():
        active_columns = [column for column in configured_columns if column in available_breakdowns]
        if configured_columns and not active_columns:
            warnings.warn(
                f"Skipped analysis level '{level_name}' because none of its breakdown columns are available.",
                stacklevel=2,
            )
            continue
        if active_columns:
            groups = working.groupby(
                [breakdown_group_columns[column] for column in active_columns],
                sort=False,
                dropna=False,
            )
            grouped_data = [
                (keys if isinstance(keys, tuple) else (keys,), group)
                for keys, group in groups
            ]
        else:
            grouped_data = [((), working)]

        for variable, analysis_type, label, list_name, relevant_chain in questions:
            excluded_columns = set(excluded_level_pairs.get(variable, ()))
            if excluded_columns.intersection(active_columns) or variable in active_columns:
                continue
            for keys, group in grouped_data:
                breakdown_values = {column: "All" for column in output_breakdowns}
                breakdown_values.update({
                    breakdown_labels.get(column, column): value
                    for column, value in zip(active_columns, keys)
                })
                weights, valid_weights = _valid_weight_mask(group, weight_col)
                base = {
                    "question_code": variable,
                    "question_label": label,
                    "question_type": {
                        "categorical_single": "categorical",
                        "multiple_response": "multiple_response",
                    }.get(analysis_type, analysis_type),
                    "analysis_level": level_name,
                    **breakdown_values,
                }
                applicable = compute_relevant_mask(group, relevant_chain)
                not_applicable_n = int((~applicable).sum())

                if analysis_type == "indicator":
                    scores = pd.to_numeric(group[variable], errors="coerce")
                    valid = scores.notna() & scores.between(0, 1)
                    if not valid.any():
                        continue
                    weighted_valid = valid & valid_weights
                    weighted_denominator = weights.loc[weighted_valid].sum()
                    weighted_score = (
                        scores.loc[weighted_valid] * weights.loc[weighted_valid]
                    ).sum()
                    rows.append({
                        **base,
                        "response_code": "indicator_score",
                        "response_label": "Indicator score",
                        "frequency": int(valid.sum()),
                        "unweighted_denominator": int(valid.sum()),
                        "weighted_frequency": (
                            round(weighted_score, decimals)
                            if weight_col in group else np.nan
                        ),
                        "weighted_denominator": (
                            round(weighted_denominator, decimals)
                            if weight_col in group else np.nan
                        ),
                        "unweighted_percentage": round(
                            scores.loc[valid].mean() * 100, decimals
                        ),
                        "weighted_percentage": (
                            round(
                                weighted_score / weighted_denominator * 100,
                                decimals,
                            )
                            if weighted_denominator > 0 else np.nan
                        ),
                        "mean": np.nan,
                        "median": np.nan,
                        "sd": np.nan,
                        "missing_n": int((~valid).sum()),
                        "not_applicable_n": 0,
                    })
                    continue

                if analysis_type == "continuous":
                    values = pd.to_numeric(group[variable], errors="coerce")
                    valid = applicable & values.notna()
                    valid_values = values.loc[valid]
                    if valid_values.empty:
                        continue
                    weighted_valid = valid & valid_weights
                    weighted_denominator = weights.loc[weighted_valid].sum()
                    rows.append({
                        **base, "response_code": pd.NA, "response_label": pd.NA,
                        "frequency": int(valid.sum()), "unweighted_denominator": int(valid.sum()),
                        "weighted_frequency": np.nan,
                        "weighted_denominator": round(weighted_denominator, decimals) if weight_col in group else np.nan,
                        "unweighted_percentage": np.nan, "weighted_percentage": np.nan,
                        "mean": round(valid_values.mean(), decimals),
                        "median": round(valid_values.median(), decimals),
                        "sd": round(valid_values.std(), decimals),
                        "missing_n": int((applicable & ~values.notna()).sum()),
                        "not_applicable_n": not_applicable_n,
                    })
                    continue

                if analysis_type == "multiple_response":
                    options = choice_maps.get(list_name, {})
                    if not options:
                        missing_multiple_choice_lists.add(variable)
                        continue
                    expected_option_columns = [f"{variable}/{code}" for code in options]
                    if variable not in group.columns or any(
                        column not in group.columns for column in expected_option_columns
                    ):
                        incomplete_multiple_response_data.add(variable)
                        continue
                    valid = applicable & group[variable].notna()
                    response_options = [
                        (str(code), str(option_label), get_multi_binary(group, variable, str(code)).astype(bool))
                        for code, option_label in options.items()
                    ]
                else:
                    values = group[variable]
                    valid = applicable & values.notna()
                    observed_codes = list(pd.Index(values.loc[valid].unique()).dropna())
                    metadata_codes = list(choice_maps.get(list_name, {}).keys())
                    response_codes = metadata_codes + [
                        code for code in observed_codes if code not in metadata_codes
                    ]
                    response_options = [
                        (
                            str(code),
                            str(display_category(code, variable, list_name_by_var, choice_maps)),
                            values.eq(code),
                        )
                        for code in response_codes
                    ]

                valid_n = int(valid.sum())
                if not valid_n or not response_options:
                    continue
                weighted_valid = valid & valid_weights
                weighted_denominator = weights.loc[weighted_valid].sum()
                missing_n = int((applicable & ~group[variable].notna()).sum())
                for response_code, response_label, selected in response_options:
                    frequency = int((selected & valid).sum())
                    weighted_frequency = weights.loc[selected & weighted_valid].sum()
                    rows.append({
                        **base, "response_code": response_code, "response_label": response_label,
                        "frequency": frequency, "unweighted_denominator": valid_n,
                        "weighted_frequency": round(weighted_frequency, decimals) if weight_col in group else np.nan,
                        "weighted_denominator": round(weighted_denominator, decimals) if weight_col in group else np.nan,
                        "unweighted_percentage": round(frequency / valid_n * 100, decimals),
                        "weighted_percentage": (
                            round(weighted_frequency / weighted_denominator * 100, decimals)
                            if weighted_denominator > 0 else np.nan
                        ),
                        "mean": np.nan, "median": np.nan, "sd": np.nan,
                        "missing_n": missing_n,
                        "not_applicable_n": not_applicable_n,
                    })

    if missing_multiple_choice_lists:
        warnings.warn(
            "Skipped multiple-response variables without compatible choice metadata: "
            f"{', '.join(sorted(missing_multiple_choice_lists))}.",
            stacklevel=2,
        )
    if incomplete_multiple_response_data:
        warnings.warn(
            "Skipped multiple-response variables without a complete set of Kobo "
            "option columns (for example, 'question/choice'): "
            f"{', '.join(sorted(incomplete_multiple_response_data))}.",
            stacklevel=2,
        )

    columns = [
        "question_code", "question_label", "question_type", "response_code",
        "response_label", "analysis_level", *output_breakdowns, "frequency",
        "unweighted_denominator", "weighted_frequency", "weighted_denominator",
        "unweighted_percentage", "weighted_percentage", "mean", "median", "sd",
        "missing_n", "not_applicable_n",
    ]
    return pd.DataFrame(rows, columns=columns)


def suppress_small_aggregate_cells(
    table: pd.DataFrame,
    *,
    minimum_n: int = 10,
) -> pd.DataFrame:
    """Mark and suppress aggregate estimates with a small observed base.

    The function uses the observed cell ``frequency`` when available and falls
    back to ``unweighted_denominator`` for outputs without a separate cell
    count. Structural labels remain visible so reviewers can see where
    suppression was applied. This mechanical rule complements rather than
    replaces a contextual disclosure-risk review.
    """
    if minimum_n < 1:
        raise ValueError("minimum_n must be at least 1.")
    output = table.copy()
    if output.empty:
        output["suppressed"] = pd.Series(dtype=bool)
        output["suppression_reason"] = pd.Series(dtype=object)
        return output

    if "frequency" in output.columns:
        base = pd.to_numeric(output["frequency"], errors="coerce")
    elif "unweighted_denominator" in output.columns:
        base = pd.to_numeric(output["unweighted_denominator"], errors="coerce")
    else:
        raise ValueError(
            "Aggregate table must include 'unweighted_denominator' or 'frequency'."
        )

    suppressed = base.notna() & base.lt(minimum_n)
    output["suppressed"] = suppressed
    output["suppression_reason"] = np.where(
        suppressed,
        f"Observed base is below the minimum disclosure threshold of {minimum_n}.",
        "",
    )
    estimate_columns = [
        "frequency",
        "weighted_frequency",
        "weighted_denominator",
        "unweighted_percentage",
        "weighted_percentage",
        "mean",
        "median",
        "sd",
        "missing_n",
        "not_applicable_n",
    ]
    for column in estimate_columns:
        if column in output.columns:
            output.loc[suppressed, column] = np.nan
    return output




def categorical_report_table(
    data: pd.DataFrame,
    variable: str,
    group_var: str,
    list_name_by_var: Mapping[str, str | None],
    choice_maps: Mapping[str, Mapping[str, str]],
    *,
    weight_col: str | None = "Sampling_Weight",
    decimals: int = 1,
    skip_self_comparison: bool = False,
    applicable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Format categorical group percentages using supplied Kobo metadata.

    The calculation and presentation are generic; callers decide whether a
    comparison is survey-specific or logically self-referential.
    """
    if skip_self_comparison:
        return pd.DataFrame()
    applicable = (
        applicable_mask.reindex(data.index, fill_value=False).astype(bool)
        if applicable_mask is not None
        else pd.Series(True, index=data.index)
    )
    result = categorical_group_percentages(
        data.loc[applicable], variable, group_var, weight_col
    )
    if result is None:
        return pd.DataFrame()
    counts, report, overall = result
    response_order = ordered_response_codes(variable, counts.columns, list_name_by_var, choice_maps)
    group_order = ordered_response_codes(group_var, counts.index, list_name_by_var, choice_maps)
    counts = counts.reindex(index=group_order, columns=response_order, fill_value=0)
    report = report.reindex(index=group_order, columns=response_order, fill_value=0).round(decimals)
    overall = overall.reindex(response_order, fill_value=0)
    display_columns = [display_category(value, variable, list_name_by_var, choice_maps) for value in response_order]
    report.index = [
        f"{display_category(group, group_var, list_name_by_var, choice_maps)} (n={int(counts.loc[group].sum())})"
        for group in group_order
    ]
    report.columns = display_columns
    total = pd.DataFrame(
        [overall.round(decimals).to_numpy()],
        index=[f"Total (n={int(counts.to_numpy().sum())})"],
        columns=display_columns,
    )
    report = pd.concat([report, total])
    report["Total"] = 100.0
    return report


def continuous_report_table(
    data: pd.DataFrame,
    variable: str,
    group_var: str,
    *,
    weight_col: str | None = "Sampling_Weight",
    applicable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Return weighted continuous summaries by one reporting breakdown."""
    if variable not in data.columns or group_var not in data.columns:
        return pd.DataFrame()
    applicable = (
        applicable_mask.reindex(data.index, fill_value=False).astype(bool)
        if applicable_mask is not None
        else pd.Series(True, index=data.index)
    )
    rows: list[pd.DataFrame] = []
    for group, subset in data.loc[applicable].groupby(group_var, dropna=False, sort=False):
        summary = continuous_descriptives(subset, variable, weight_col=weight_col)
        if summary.empty:
            continue
        summary.index = ["Missing" if pd.isna(group) else str(group)]
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows)
    output.index.name = "Group"
    return output


def multi_response_report_table(
    data: pd.DataFrame,
    variable: str,
    group_var: str,
    list_name: str | None,
    choice_maps: Mapping[str, Mapping[str, str]],
    *,
    weight_col: str | None = "Sampling_Weight",
    applicable_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one long option-by-group multiple-response table."""
    if variable not in data.columns or group_var not in data.columns:
        return pd.DataFrame()
    applicable = (
        applicable_mask.reindex(data.index, fill_value=False).astype(bool)
        if applicable_mask is not None
        else pd.Series(True, index=data.index)
    )
    rows: list[pd.DataFrame] = []
    for group, subset in data.loc[applicable].groupby(group_var, dropna=False, sort=False):
        table = multi_response_descriptives(
            subset,
            variable,
            list_name,
            choice_maps,
            weight_col=weight_col,
        )
        if table.empty:
            continue
        table.insert(0, "Group", "Missing" if pd.isna(group) else str(group))
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).set_index(["Group", "Option"])


def indicator_report_table(
    data: pd.DataFrame,
    indicator_var: str,
    group_var: str,
    *,
    weight_col: str | None = "Sampling_Weight",
    decimals: int = 1,
) -> pd.DataFrame:
    """Summarize a binary or 0–1 composite indicator by reporting group."""
    if indicator_var not in data.columns or group_var not in data.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group, subset in data.groupby(group_var, dropna=False, sort=False):
        scores = pd.to_numeric(subset[indicator_var], errors="coerce")
        valid = scores.notna() & scores.between(0, 1)
        if not valid.any():
            continue
        row: dict[str, Any] = {
            "Group": "Missing" if pd.isna(group) else str(group),
            "N": int(valid.sum()),
            "Unweighted_Percent": round(scores.loc[valid].mean() * 100, decimals),
            "Missing_N": int((~valid).sum()),
        }
        if weight_col and weight_col in subset.columns:
            weights = pd.to_numeric(subset[weight_col], errors="coerce")
            weighted_valid = valid & weights.notna() & (weights >= 0)
            denominator = weights.loc[weighted_valid].sum()
            row["Weighted_Percent"] = (
                round(
                    (scores.loc[weighted_valid] * weights.loc[weighted_valid]).sum()
                    / denominator
                    * 100,
                    decimals,
                )
                if denominator > 0
                else np.nan
            )
            row["Weighted_Denominator"] = round(float(denominator), decimals)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Group")
