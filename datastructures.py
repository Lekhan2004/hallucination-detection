# datastructures.py

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional
import re

# Regex patterns used across modules
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
GSM_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
FINAL_ANSWER_RE = re.compile(r"Final answer\s*[:=-]\s*([^\n]+)", re.IGNORECASE)


@dataclass
class ProblemExample:
    id: str
    dataset: str
    split: str
    problem: str
    gold_answer_raw: str
    gold_answer_norm: str


@dataclass
class ModelAnswer:
    id: str
    dataset: str
    split: str
    model: str
    problem: str
    gold_answer_raw: str
    gold_answer_norm: str
    our_final_answer_raw: Optional[str]
    our_final_answer_norm: Optional[str]
    full_response: str
    is_correct: Optional[bool]


def strip_latex_wrappers(ans: str) -> str:
    """Remove simple LaTeX wrappers like \\boxed{} and inline $...$."""
    ans = BOXED_RE.sub(r"\1", ans)
    ans = ans.strip()
    if ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1]
    return ans.strip()


def normalize_answer_str(ans: str) -> str:
    """Normalize answer string for comparison."""
    ans = ans.strip()
    ans = strip_latex_wrappers(ans)
    if ans.endswith("."):
        ans = ans[:-1]
    ans = ans.replace(",", "")
    ans = re.sub(r"\s+", " ", ans)
    return ans.strip()


def parse_numeric(ans: str) -> Optional[Fraction]:
    """
    Try to parse ans into a Fraction.
    Supports integers, decimals, and a/b style fractions.
    Returns None if parsing fails.
    """
    ans = normalize_answer_str(ans)

    # Fraction pattern a/b
    frac_match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", ans)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den == 0:
            return None
        return Fraction(num, den)

    # Decimal or integer
    num_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)", ans)
    if num_match:
        return Fraction(num_match.group(1))

    return None


def answers_equal(gold: str, pred: str) -> bool:
    """Compare gold and predicted answers in a math aware way."""
    gold_norm = normalize_answer_str(gold)
    pred_norm = normalize_answer_str(pred)

    g_num = parse_numeric(gold_norm)
    p_num = parse_numeric(pred_norm)

    if g_num is not None and p_num is not None:
        return g_num == p_num

    return gold_norm == pred_norm
