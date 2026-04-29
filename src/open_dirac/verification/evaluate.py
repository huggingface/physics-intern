"""Answer evaluation for one-shot LLM responses.

Compares an LLM's answer against the known answer from the problem YAML.
Supports symbolic (SymPy) and numerical (float) answer types.

Symbolic comparison uses a cascading chain:
  1. simplify(candidate - truth) == 0
  2. simplify(expand(candidate - truth)) == 0
  3. simplify(candidate / truth) == 1  (ratio test)
  4. candidate.equals(truth)  (random numerical substitution)
  5. Numerical fallback for constant expressions
"""

from __future__ import annotations

import re
import signal
import traceback
from typing import Any

_SYMPY_TIMEOUT = 30  # seconds per comparison step


class _SympyTimeout(Exception):
    """Raised when a SymPy computation exceeds the time limit."""


def _with_timeout(fn, timeout=_SYMPY_TIMEOUT):
    """Run *fn*() with a hard timeout using SIGALRM.

    Thread-based timeouts (ThreadPoolExecutor) cannot reliably interrupt
    CPU-bound SymPy computations because the GIL prevents the main thread
    from checking the timeout while SymPy runs tight Python loops (e.g.
    ``.equals()`` on Quantity expressions with symbolic exponents).

    SIGALRM is delivered asynchronously by the OS and interrupts the
    computation regardless of GIL state.
    """

    def _alarm_handler(signum, frame):
        raise _SympyTimeout(f"Computation exceeded {timeout}s timeout")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        return fn()
    except _SympyTimeout:
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_answer_code(response_text: str) -> str | None:
    """Return the last ``python code block containing ``def answer`` in *response_text*."""
    pattern = r"```python\s*\n(.*?)```"
    blocks = re.findall(pattern, response_text, re.DOTALL)
    for block in reversed(blocks):
        if "def answer" in block:
            return block.strip()
    return None


# ---------------------------------------------------------------------------
# Answer type classification
# ---------------------------------------------------------------------------


def _classify_answer_type(answer_value: Any) -> str:
    """Return ``"numerical"`` if *answer_value* is a plain number, else ``"symbolic"``."""
    if isinstance(answer_value, (int, float)):
        return "numerical"
    try:
        float(str(answer_value).strip())
        return "numerical"
    except (ValueError, TypeError):
        return "symbolic"


# ---------------------------------------------------------------------------
# Template parsing
# ---------------------------------------------------------------------------


def _parse_template_preamble(answer_template: str) -> tuple[str, list[str]]:
    """Extract preamble code and ``answer()`` parameter names from *answer_template*.

    The preamble is everything before the ``def answer(...)`` line —
    typically ``import sympy`` and symbol definitions.
    """
    lines = answer_template.strip().splitlines()
    preamble_lines: list[str] = []
    param_names: list[str] = []
    for line in lines:
        if line.strip().startswith("def answer("):
            match = re.match(r"def answer\(([^)]*)\)", line.strip())
            if match and match.group(1).strip():
                param_names = [p.strip() for p in match.group(1).split(",")]
            break
        preamble_lines.append(line)
    return "\n".join(preamble_lines), param_names


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------


def _compare_numerical(candidate: Any, truth: float, rtol: float = 0.01) -> dict:
    """Compare numerical answers with relative tolerance."""
    try:
        candidate_val = float(candidate)
    except (TypeError, ValueError) as exc:
        return {
            "correct": None,
            "method": "numerical_conversion_error",
            "error": f"Cannot convert candidate to float: {exc}",
            "details": "",
        }
    if truth == 0:
        correct = abs(candidate_val) < 1e-10
    else:
        correct = abs(candidate_val - truth) / abs(truth) < rtol
    return {
        "correct": correct,
        "method": "numerical",
        "error": None,
        "details": f"candidate={candidate_val}, truth={truth}, rtol={rtol}",
    }


# ---------------------------------------------------------------------------
# Symbolic comparison
# ---------------------------------------------------------------------------


def _compare_single_symbolic(candidate: Any, truth: Any) -> dict:
    """Compare a single candidate value against a single truth value.

    Handles string equality for discrete answers (e.g. multiple-choice)
    and the cascading SymPy simplification chain for symbolic expressions.
    """
    # String comparison (e.g. multiple-choice answers like 'C')
    if isinstance(truth, str):
        correct = str(candidate).strip() == truth.strip()
        return {
            "correct": correct,
            "method": "string_equality",
            "error": None,
            "details": f"candidate={candidate!r}, truth={truth!r}",
        }

    import sympy as sp

    # --- Cascading comparison chain ---

    # 1. simplify(candidate - truth) == 0
    try:
        diff = _with_timeout(lambda: sp.simplify(candidate - truth))
        if diff == 0:
            return {
                "correct": True,
                "method": "simplify",
                "error": None,
                "details": str(diff),
            }
    except (_SympyTimeout, Exception):
        pass

    # 2. simplify(expand(candidate - truth)) == 0
    try:
        diff = _with_timeout(lambda: sp.simplify(sp.expand(candidate - truth)))
        if diff == 0:
            return {
                "correct": True,
                "method": "expand_simplify",
                "error": None,
                "details": str(diff),
            }
    except (_SympyTimeout, Exception):
        pass

    # 3. Ratio test: simplify(candidate / truth) == 1
    try:
        ratio = _with_timeout(lambda: sp.simplify(candidate / truth))
        if ratio == 1:
            return {
                "correct": True,
                "method": "ratio",
                "error": None,
                "details": str(ratio),
            }
    except (_SympyTimeout, Exception):
        pass

    # 4. .equals() — random numerical substitution (most robust)
    try:
        if _with_timeout(lambda: candidate.equals(truth)):
            return {
                "correct": True,
                "method": "equals",
                "error": None,
                "details": "candidate.equals(truth)",
            }
    except (_SympyTimeout, Exception):
        pass

    # 5. Numerical fallback for constant expressions (no free symbols)
    try:
        c_val = complex(candidate)
        t_val = complex(truth)
        if t_val != 0 and abs(c_val - t_val) / abs(t_val) < 1e-9:
            return {
                "correct": True,
                "method": "numerical_fallback",
                "error": None,
                "details": f"candidate={c_val}, truth={t_val}",
            }
    except Exception:
        pass

    return {
        "correct": False,
        "method": "all_methods_failed",
        "error": None,
        "details": f"candidate={candidate}, truth={truth}",
    }


def _eval_truth(answer_str: str, namespace: dict[str, Any]) -> tuple[Any, str | None]:
    """Evaluate a truth answer string into a value.

    Returns (truth_value, error_string).  *error_string* is None on success.
    For multi-line answer blocks (reference files with intermediate variable
    definitions), ``exec()`` the block and collect assigned variables.  If the
    candidate is a tuple, the last N assigned variables form the truth tuple.
    """
    truth_ns = dict(namespace)

    if "\n" not in answer_str:
        # Single-line: eval the RHS
        rhs = (
            answer_str.split("=", 1)[1].strip()
            if "=" in answer_str
            else answer_str.strip()
        )
        try:
            return eval(rhs, truth_ns), None  # noqa: S307
        except Exception as exc:
            return None, f"Truth expression eval failed: {exc}"

    # Multi-line block: exec all lines and collect assigned variable names
    try:
        exec(answer_str, truth_ns)  # noqa: S102
    except Exception as exc:
        return None, f"Truth expression eval failed: {exc}"

    var_names: list[str] = []
    for line in answer_str.strip().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            lhs = stripped.split("=", 1)[0].strip()
            if lhs.isidentifier() and lhs in truth_ns:
                var_names.append(lhs)
    if not var_names:
        return None, "No assignments found in multi-line answer"
    return {name: truth_ns[name] for name in var_names}, None


def _compare_symbolic(
    candidate: Any,
    answer_str: str,
    namespace: dict[str, Any],
) -> dict:
    """Compare symbolic answers using cascading simplification methods."""
    truth_val, err = _eval_truth(answer_str, namespace)
    if err is not None:
        return {
            "correct": None,
            "method": "truth_eval_error",
            "error": err,
            "details": traceback.format_exc(),
        }

    # Multi-line block returns a dict of assigned variables
    if isinstance(truth_val, dict):
        truth_vars = truth_val  # {name: value, ...}
        if isinstance(candidate, (tuple, list)):
            # Take the last N variables to skip intermediates
            names = list(truth_vars.keys())
            if len(candidate) <= len(names):
                names = names[-len(candidate) :]
            truth_values = [truth_vars[n] for n in names]
        else:
            # Single candidate: use the last variable
            truth_values = [list(truth_vars.values())[-1]]
            candidate = [candidate]

        # Element-by-element comparison
        element_results = []
        all_correct = True
        for cand_elem, truth_elem in zip(candidate, truth_values):
            r = _compare_single_symbolic(cand_elem, truth_elem)
            element_results.append(r)
            if not r["correct"]:
                all_correct = False

        methods = [r["method"] for r in element_results]
        details = "; ".join(
            f"[{i}] {r['method']}: {r['details']}"
            for i, r in enumerate(element_results)
        )
        return {
            "correct": all_correct,
            "method": f"multiline({','.join(methods)})",
            "error": None,
            "details": details,
        }

    # Single truth value — standard cascading comparison
    return _compare_single_symbolic(candidate, truth_val)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _compare_tuple(candidate: Any, truth: Any) -> dict:
    """Compare candidate and truth values element-by-element.

    Both *candidate* and *truth* may be scalars or tuples/lists.  Each element
    pair is compared via ``_compare_single_symbolic``.
    """
    cand_list = list(candidate) if isinstance(candidate, (tuple, list)) else [candidate]
    truth_list = list(truth) if isinstance(truth, (tuple, list)) else [truth]

    if len(cand_list) != len(truth_list):
        return {
            "correct": False,
            "method": "shape_mismatch",
            "error": f"Candidate returns {len(cand_list)} values, truth returns {len(truth_list)}",
            "details": "",
        }

    element_results = []
    all_correct = True
    for cand_elem, truth_elem in zip(cand_list, truth_list):
        r = _compare_single_symbolic(cand_elem, truth_elem)
        element_results.append(r)
        if not r["correct"]:
            all_correct = False

    if len(element_results) == 1:
        return element_results[0]

    methods = [r["method"] for r in element_results]
    details = "; ".join(
        f"[{i}] {r['method']}: {r['details']}" for i, r in enumerate(element_results)
    )
    return {
        "correct": all_correct,
        "method": f"reference_fn({','.join(methods)})",
        "error": None,
        "details": details,
    }


def evaluate_response(
    response_text: str,
    problem_def: dict,
    *,
    reference_code: str | None = None,
) -> dict:
    """Evaluate an LLM response against the known answer from *problem_def*.

    If *reference_code* is provided (a string containing a ``def answer(...)``
    function), the truth value is obtained by executing that function in the
    template preamble namespace — bypassing string-based answer parsing.

    Returns a dict with keys:
        correct  — True / False / None (if evaluation errored)
        method   — which comparison method succeeded (or error label)
        error    — error message string, or None
        details  — human-readable extra info
    """
    answer_value = problem_def.get("answer")
    answer_template = problem_def.get("answer_template", "")

    has_reference_fn = reference_code is not None
    if not has_reference_fn:
        if answer_value is None or (
            isinstance(answer_value, str) and not answer_value.strip()
        ):
            return {
                "correct": None,
                "method": "no_answer",
                "error": "No answer in problem definition",
                "details": "",
            }

    answer_type = (
        _classify_answer_type(answer_value) if not has_reference_fn else "symbolic"
    )
    answer_str = str(answer_value).strip() if not has_reference_fn else ""

    # Extract candidate code from LLM response
    candidate_code = extract_answer_code(response_text)
    if candidate_code is None:
        return {
            "correct": None,
            "method": "no_code",
            "error": "No python code block with def answer found in response",
            "details": "",
        }

    # Parse template preamble for shared namespace
    preamble, param_names = _parse_template_preamble(answer_template)

    # Build shared namespace from preamble
    namespace: dict[str, Any] = {}
    try:
        if preamble.strip():
            exec(preamble, namespace)  # noqa: S102
    except Exception as exc:
        return {
            "correct": None,
            "method": "preamble_error",
            "error": f"Preamble exec failed: {exc}",
            "details": traceback.format_exc(),
        }

    # Ensure sympy is available
    if "sp" not in namespace:
        try:
            import sympy as sp

            namespace["sp"] = sp
        except ImportError:
            pass

    # Execute candidate code in the shared namespace
    candidate_ns = dict(namespace)
    try:
        exec(candidate_code, candidate_ns)  # noqa: S102
    except Exception as exc:
        return {
            "correct": None,
            "method": "candidate_exec_error",
            "error": f"Candidate code exec failed: {exc}",
            "details": traceback.format_exc(),
        }

    if "answer" not in candidate_ns or not callable(candidate_ns["answer"]):
        return {
            "correct": None,
            "method": "no_answer_func",
            "error": "Candidate code has no callable answer()",
            "details": "",
        }

    # Call candidate's answer() with canonical symbols from preamble
    try:
        if param_names:
            args = [candidate_ns[name] for name in param_names]
            candidate_result = candidate_ns["answer"](*args)
        else:
            candidate_result = candidate_ns["answer"]()
    except Exception as exc:
        return {
            "correct": None,
            "method": "candidate_call_error",
            "error": f"answer() call failed: {exc}",
            "details": traceback.format_exc(),
        }

    # --- Reference function path: exec + call + direct comparison ---
    if has_reference_fn:
        ref_ns = dict(namespace)
        try:
            exec(reference_code, ref_ns)  # noqa: S102
        except Exception as exc:
            return {
                "correct": None,
                "method": "reference_exec_error",
                "error": f"Reference code exec failed: {exc}",
                "details": traceback.format_exc(),
            }
        if "answer" not in ref_ns or not callable(ref_ns["answer"]):
            return {
                "correct": None,
                "method": "reference_error",
                "error": "Reference code has no callable answer()",
                "details": "",
            }
        try:
            if param_names:
                ref_args = [ref_ns[name] for name in param_names]
                truth_result = ref_ns["answer"](*ref_args)
            else:
                truth_result = ref_ns["answer"]()
        except Exception as exc:
            return {
                "correct": None,
                "method": "reference_call_error",
                "error": f"Reference answer() call failed: {exc}",
                "details": traceback.format_exc(),
            }
        return _compare_tuple(candidate_result, truth_result)

    # --- String-based comparison (original path) ---
    if answer_type == "numerical":
        return _compare_numerical(candidate_result, float(answer_str))
    else:
        return _compare_symbolic(candidate_result, answer_str, namespace)
