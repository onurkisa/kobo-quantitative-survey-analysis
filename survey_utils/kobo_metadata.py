"""Generic helpers for Kobo/XLSForm metadata and response exports."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def build_choice_maps(choices_df: pd.DataFrame, label_col: str) -> dict[str, dict[str, str]]:
    """Build canonical choice-code-to-label mappings from an XLSForm choices sheet.

    An absent or empty choices sheet is a valid survey feature and produces an
    empty mapping. A non-empty choices sheet must still contain `list_name` and
    `name`, because its structure is otherwise malformed.
    """
    if choices_df.empty:
        return {}
    required_columns = {"list_name", "name"}
    missing_columns = required_columns - set(choices_df.columns)
    if missing_columns:
        raise ValueError(
            "The XLSForm choices sheet is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    label_values = choices_df[label_col] if label_col in choices_df.columns else choices_df["name"]
    choices = choices_df.assign(_display_label=label_values)
    return {
        list_name: dict(zip(group["name"].astype(str), group["_display_label"].astype(str)))
        for list_name, group in choices.groupby("list_name")
    }


def parse_survey_structure(survey_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Extract question metadata, hierarchy, and inherited relevant logic.

    Args:
        survey_df: Kobo XLSForm `survey` worksheet.
        label_col: Preferred analyst-facing label column.

    Returns:
        One row per non-structural XLSForm question. Question names are used as
        labels when the requested label column is absent.
    """
    required_columns = {"type", "name"}
    missing_columns = required_columns - set(survey_df.columns)
    if missing_columns:
        raise ValueError(
            "The XLSForm survey sheet is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    records: list[dict[str, Any]] = []
    section_stack: list[Any] = []
    relevant_stack: list[Any] = []
    has_label_column = label_col in survey_df.columns

    for _, row in survey_df.iterrows():
        question_type = str(row["type"]).strip()
        name = row["name"]
        label = row[label_col] if has_label_column else name
        own_relevant = row.get("relevant")
        own_relevant = own_relevant if pd.notna(own_relevant) else None

        if question_type in ("begin_group", "begin_repeat"):
            section_label = label if pd.notna(label) else name
            if question_type == "begin_repeat":
                section_label = f"[repeat] {section_label}"
            section_stack.append(section_label)
            relevant_stack.append(own_relevant)
            continue

        if question_type in ("end_group", "end_repeat"):
            if section_stack:
                section_stack.pop()
            if relevant_stack:
                relevant_stack.pop()
            continue

        if question_type in ("start", "end", "deviceid"):
            continue
        if pd.isna(name) or not str(name).strip():
            continue

        relevant_chain = [item for item in relevant_stack if item]
        if own_relevant:
            relevant_chain.append(own_relevant)
        records.append(
            {
                "name": name,
                "type": question_type,
                "label": label,
                "section": section_stack[-1] if section_stack else "Root",
                "relevant_chain": relevant_chain,
                "required": _as_xlsform_boolean(row.get("required")),
            }
        )
    return pd.DataFrame(records)


def classify_question(question_type: str) -> str:
    """Return the analysis category associated with an XLSForm question type."""
    if question_type.startswith("select_one") and "_from_file" not in question_type:
        return "categorical_single"
    if question_type.startswith("select_multiple"):
        return "multiple_response"
    if question_type in ("integer", "decimal"):
        return "continuous"
    if question_type == "text":
        return "open_text"
    return "other"


def get_list_name(question_type: str) -> str | None:
    """Return the choice-list name referenced by an XLSForm question type."""
    parts = question_type.split()
    return parts[1] if len(parts) > 1 else None


def _as_xlsform_boolean(value: Any) -> bool:
    """Interpret common XLSForm truthy values without treating ``"no"`` as true."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1", "required"}


class _UnsupportedRelevant(ValueError):
    """Raised internally when a relevance expression uses unsupported syntax."""


_TOKEN_PATTERN = re.compile(
    r"\s*(?:"
    r"(?P<SELECTED>selected\(\s*\$\{[A-Za-z_][\w.:-]*\}\s*,\s*(?:'[^']*'|\"[^\"]*\")\s*\))|"
    r"(?P<VAR>\$\{[A-Za-z_][\w.:-]*\})|"
    r"(?P<NUMBER>-?(?:\d+(?:\.\d*)?|\.\d+))|"
    r"(?P<STRING>'[^']*'|\"[^\"]*\")|"
    r"(?P<OP>!=|<=|>=|=|<|>)|"
    r"(?P<LPAREN>\()|(?P<RPAREN>\))|"
    r"(?P<BOOL>and\b|or\b|not\b)"
    r")",
    flags=re.IGNORECASE,
)


def _tokenize_relevant(expression: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(expression):
        match = _TOKEN_PATTERN.match(expression, position)
        if not match:
            raise _UnsupportedRelevant(
                f"unsupported syntax near {expression[position:position + 30]!r}"
            )
        kind = match.lastgroup
        if kind is None:
            raise _UnsupportedRelevant("unrecognised expression token")
        value = match.group(kind)
        tokens.append((kind, value.lower() if kind == "BOOL" else value))
        position = match.end()
    return tokens


class _RelevantParser:
    def __init__(self, data: pd.DataFrame, tokens: list[tuple[str, str]]):
        self.data = data
        self.tokens = tokens
        self.position = 0

    def parse(self) -> pd.Series:
        result = self._parse_or()
        if self.position != len(self.tokens):
            raise _UnsupportedRelevant("unexpected trailing tokens")
        return result

    def _peek(self, kind: str, value: str | None = None) -> bool:
        if self.position >= len(self.tokens):
            return False
        token_kind, token_value = self.tokens[self.position]
        return token_kind == kind and (value is None or token_value == value)

    def _take(self, kind: str, value: str | None = None) -> str:
        if not self._peek(kind, value):
            raise _UnsupportedRelevant(f"expected {value or kind}")
        token = self.tokens[self.position][1]
        self.position += 1
        return token

    def _parse_or(self) -> pd.Series:
        result = self._parse_and()
        while self._peek("BOOL", "or"):
            self._take("BOOL", "or")
            result = result | self._parse_and()
        return result

    def _parse_and(self) -> pd.Series:
        result = self._parse_not()
        while self._peek("BOOL", "and"):
            self._take("BOOL", "and")
            result = result & self._parse_not()
        return result

    def _parse_not(self) -> pd.Series:
        if self._peek("BOOL", "not"):
            self._take("BOOL", "not")
            return ~self._parse_not()
        return self._parse_primary()

    def _parse_primary(self) -> pd.Series:
        if self._peek("LPAREN"):
            self._take("LPAREN")
            result = self._parse_or()
            self._take("RPAREN")
            return result
        if self._peek("SELECTED"):
            return self._selected(self._take("SELECTED"))
        variable_token = self._take("VAR")
        operator = self._take("OP")
        if self._peek("STRING"):
            literal: Any = self._take("STRING")[1:-1]
        elif self._peek("NUMBER"):
            literal = float(self._take("NUMBER"))
        else:
            raise _UnsupportedRelevant("comparison requires a string or number")
        variable = variable_token[2:-1]
        if variable not in self.data.columns:
            raise _UnsupportedRelevant(f"referenced variable {variable!r} is absent")
        series = self.data[variable]
        if isinstance(literal, float):
            left = pd.to_numeric(series, errors="coerce")
        else:
            left = series.astype("string")
        operations = {
            "=": lambda: left.eq(literal),
            "!=": lambda: left.ne(literal),
            "<": lambda: left.lt(literal),
            "<=": lambda: left.le(literal),
            ">": lambda: left.gt(literal),
            ">=": lambda: left.ge(literal),
        }
        return operations[operator]().fillna(False)

    def _selected(self, token: str) -> pd.Series:
        match = re.fullmatch(
            r"selected\(\s*\$\{([A-Za-z_][\w.:-]*)\}\s*,\s*(['\"])(.*?)\2\s*\)",
            token,
            flags=re.IGNORECASE,
        )
        if not match:
            raise _UnsupportedRelevant("malformed selected() expression")
        variable, _, value = match.groups()
        exploded_column = f"{variable}/{value}"
        if exploded_column in self.data.columns:
            return pd.to_numeric(
                self.data[exploded_column], errors="coerce"
            ).fillna(0).eq(1)
        if variable not in self.data.columns:
            raise _UnsupportedRelevant(f"referenced variable {variable!r} is absent")
        return self.data[variable].astype("string").fillna("").map(
            lambda item: value in str(item).split()
        )


def _relevant_expr_mask(data: pd.DataFrame, expression: str | None) -> pd.Series:
    """Evaluate common XLSForm relevance syntax, warning on unsupported input."""
    if not expression or pd.isna(expression):
        return pd.Series(True, index=data.index)
    try:
        return _RelevantParser(data, _tokenize_relevant(str(expression))).parse()
    except _UnsupportedRelevant as exc:
        warnings.warn(
            f"Unsupported relevant expression {expression!r}: {exc}. "
            "The question is treated as applicable to avoid false exclusions.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series(True, index=data.index)


def compute_relevant_mask(data: pd.DataFrame, relevant_chain: Sequence[str] | None) -> pd.Series:
    """Return records to which a question's complete relevant chain applies."""
    mask = pd.Series(True, index=data.index)
    if not isinstance(relevant_chain, Sequence) or isinstance(relevant_chain, str):
        return mask
    for expression in relevant_chain:
        mask &= _relevant_expr_mask(data, expression)
    return mask


def normalize_choice_value(value: Any) -> Any:
    """Normalize apostrophes, whitespace, and case for choice matching."""
    if not isinstance(value, str):
        return value
    return value.replace("\u2019", "'").replace("\u2018", "'").strip().lower()


def build_choice_lookup(choice_maps: Mapping[str, Mapping[str, str]], list_name: str | None) -> dict[Any, str]:
    """Map normalized choice labels and codes to canonical choice codes."""
    lookup: dict[Any, str] = {}
    for code, label in choice_maps.get(list_name, {}).items():
        lookup[normalize_choice_value(label)] = code
        lookup[normalize_choice_value(code)] = code
    return lookup


def ordered_response_codes(
    variable: str,
    observed_values: Sequence[Any],
    list_name_by_var: Mapping[str, str | None],
    choice_maps: Mapping[str, Mapping[str, str]],
) -> list[Any]:
    """Return metadata choice order followed by observed non-metadata values."""
    list_name = list_name_by_var.get(variable)
    choice_order = list(choice_maps.get(list_name, {}).keys())
    observed = list(pd.Index(observed_values).dropna())
    return [value for value in choice_order if value in observed] + [
        value for value in observed if value not in choice_order
    ]


def display_category(
    value: Any,
    variable: str,
    list_name_by_var: Mapping[str, str | None],
    choice_maps: Mapping[str, Mapping[str, str]],
) -> Any:
    """Return a metadata label when available, otherwise preserve the value."""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    return choice_maps.get(list_name_by_var.get(variable), {}).get(str(value), value)


def get_question_label(qmeta: pd.DataFrame, variable: str) -> str:
    """Return a metadata question label, with a readable variable-name fallback."""
    if {"name", "label"}.issubset(qmeta.columns):
        row = qmeta.loc[qmeta["name"] == variable, "label"]
        if not row.empty and pd.notna(row.iloc[0]):
            return str(row.iloc[0])
    return variable.replace("_", " ")


def get_multi_binary(data: pd.DataFrame, variable: str, option_code: str) -> pd.Series:
    """Return a Kobo multiple-response option as a binary selection indicator."""
    if variable not in data.columns:
        return pd.Series(0, index=data.index, dtype=int)
    exploded_column = f"{variable}/{option_code}"
    if exploded_column in data.columns:
        return pd.to_numeric(data[exploded_column], errors="coerce").fillna(0).astype(int)
    return data[variable].astype(str).str.contains(re.escape(option_code), na=False).astype(int)
