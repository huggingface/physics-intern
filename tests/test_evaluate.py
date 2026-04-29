"""Tests for open_dirac.verification.evaluate — answer evaluation module."""

import sympy as sp

from open_dirac.verification.evaluate import (
    evaluate_response,
    extract_answer_code,
    _classify_answer_type,
    _compare_numerical,
    _compare_symbolic,
    _parse_template_preamble,
)


# ── extract_answer_code ────────────────────────────────────────────────────


class TestExtractAnswerCode:
    def test_single_block(self):
        text = "Some explanation\n```python\ndef answer(x):\n    return x\n```\nDone."
        assert extract_answer_code(text) == "def answer(x):\n    return x"

    def test_multiple_blocks_returns_last_with_def_answer(self):
        text = (
            "```python\nprint('hello')\n```\n"
            "```python\ndef answer():\n    return 42\n```\n"
        )
        assert "return 42" in extract_answer_code(text)

    def test_ignores_blocks_without_def_answer(self):
        text = "```python\nx = 1\n```\n"
        assert extract_answer_code(text) is None

    def test_no_code_blocks(self):
        assert extract_answer_code("Just text, no code.") is None

    def test_last_block_wins(self):
        text = (
            "```python\ndef answer():\n    return 1\n```\n"
            "Some text\n"
            "```python\ndef answer():\n    return 2\n```\n"
        )
        code = extract_answer_code(text)
        assert "return 2" in code


# ── _classify_answer_type ──────────────────────────────────────────────────


class TestClassifyAnswerType:
    def test_float(self):
        assert _classify_answer_type(1057.8) == "numerical"

    def test_int(self):
        assert _classify_answer_type(42) == "numerical"

    def test_float_string(self):
        assert _classify_answer_type("0.7687") == "numerical"

    def test_sympy_expression_string(self):
        assert _classify_answer_type("hbar * c / (8 * sp.pi * G)") == "symbolic"

    def test_assignment_string(self):
        assert (
            _classify_answer_type("T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)")
            == "symbolic"
        )


# ── _parse_template_preamble ──────────────────────────────────────────────


class TestParseTemplatePreamble:
    def test_with_params(self):
        template = (
            "import sympy as sp\n"
            "\n"
            "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n"
            "\n"
            "def answer(hbar, c, G, M, k_B):\n"
            "    T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)\n"
            "    return T_H\n"
        )
        preamble, params = _parse_template_preamble(template)
        assert params == ["hbar", "c", "G", "M", "k_B"]
        assert "import sympy" in preamble
        assert "sp.symbols" in preamble
        assert "def answer" not in preamble

    def test_no_params(self):
        template = "def answer():\n    return 1057.8\n"
        preamble, params = _parse_template_preamble(template)
        assert params == []
        assert preamble.strip() == ""


# ── _compare_numerical ────────────────────────────────────────────────────


class TestCompareNumerical:
    def test_exact_match(self):
        result = _compare_numerical(1057.8, 1057.8)
        assert result["correct"] is True

    def test_close_match(self):
        result = _compare_numerical(1058.0, 1057.8)
        assert result["correct"] is True

    def test_wrong_answer(self):
        result = _compare_numerical(500.0, 1057.8)
        assert result["correct"] is False

    def test_zero_truth(self):
        result = _compare_numerical(0.0, 0.0)
        assert result["correct"] is True

    def test_non_convertible(self):
        result = _compare_numerical("not_a_number", 1.0)
        assert result["correct"] is None


# ── _compare_symbolic ─────────────────────────────────────────────────────


class TestCompareSymbolic:
    def _make_namespace(self):
        ns = {}
        exec("import sympy as sp", ns)
        exec("hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)", ns)
        return ns

    def test_identical_expressions(self):
        ns = self._make_namespace()
        hbar, c, G, M, k_B = ns["hbar"], ns["c"], ns["G"], ns["M"], ns["k_B"]
        candidate = hbar * c**3 / (8 * sp.pi * G * M * k_B)
        answer_str = "T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)"
        result = _compare_symbolic(candidate, answer_str, ns)
        assert result["correct"] is True
        assert result["method"] == "simplify"

    def test_equivalent_form(self):
        """Algebraically equivalent but written differently."""
        ns = self._make_namespace()
        hbar, c, G, M, k_B = ns["hbar"], ns["c"], ns["G"], ns["M"], ns["k_B"]
        # Factor out differently
        candidate = c**3 * hbar / (8 * sp.pi * G * k_B * M)
        answer_str = "T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)"
        result = _compare_symbolic(candidate, answer_str, ns)
        assert result["correct"] is True

    def test_wrong_coefficient(self):
        ns = self._make_namespace()
        hbar, c, G, M, k_B = ns["hbar"], ns["c"], ns["G"], ns["M"], ns["k_B"]
        # Wrong: factor of 4 instead of 8
        candidate = hbar * c**3 / (4 * sp.pi * G * M * k_B)
        answer_str = "T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)"
        result = _compare_symbolic(candidate, answer_str, ns)
        assert result["correct"] is False

    def test_ratio_test(self):
        """Expression where simplify(diff)==0 may fail but ratio works."""
        ns = {}
        exec("import sympy as sp", ns)
        exec("x = sp.Symbol('x', positive=True)", ns)
        x = ns["x"]
        # Two equivalent forms: x*(x+1)/x vs (x+1)
        candidate = x * (x + 1) / x
        answer_str = "x + 1"
        result = _compare_symbolic(candidate, answer_str, ns)
        assert result["correct"] is True

    def test_equals_fallback(self):
        """Test .equals() path with trig identities."""
        ns = {}
        exec("import sympy as sp", ns)
        exec("x = sp.Symbol('x')", ns)
        x = ns["x"]
        candidate = sp.sin(x) ** 2 + sp.cos(x) ** 2
        answer_str = "1"
        result = _compare_symbolic(candidate, answer_str, ns)
        assert result["correct"] is True

    def test_rational_answer(self):
        """sp.Rational answer (like tov_buchdahl)."""
        ns = {}
        exec("import sympy as sp", ns)
        candidate = sp.Rational(8, 9)
        answer_str = "sp.Rational(8, 9)"
        result = _compare_symbolic(candidate, answer_str, ns)
        assert result["correct"] is True

    def test_truth_eval_error(self):
        ns = {}
        exec("import sympy as sp", ns)
        result = _compare_symbolic(sp.Integer(1), "undefined_var + 1", ns)
        assert result["correct"] is None
        assert result["method"] == "truth_eval_error"


# ── evaluate_response (full pipeline) ─────────────────────────────────────


class TestEvaluateResponse:
    def test_correct_hawking(self):
        """Full pipeline with Hawking temperature."""
        problem_def = {
            "answer": "T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)\n",
            "answer_template": (
                "import sympy as sp\n\n"
                "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
                "def answer(hbar, c, G, M, k_B):\n"
                "    T_H = ...\n"
                "    return T_H\n"
            ),
        }
        response = (
            "The Hawking temperature is...\n\n"
            "```python\n"
            "import sympy as sp\n\n"
            "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
            "def answer(hbar, c, G, M, k_B):\n"
            "    T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)\n"
            "    return T_H\n"
            "```\n"
        )
        result = evaluate_response(response, problem_def)
        assert result["correct"] is True

    def test_incorrect_hawking(self):
        """Wrong coefficient in Hawking temperature."""
        problem_def = {
            "answer": "T_H = hbar * c**3 / (8 * sp.pi * G * M * k_B)\n",
            "answer_template": (
                "import sympy as sp\n\n"
                "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
                "def answer(hbar, c, G, M, k_B):\n"
                "    T_H = ...\n"
                "    return T_H\n"
            ),
        }
        response = (
            "```python\n"
            "import sympy as sp\n\n"
            "hbar, c, G, M, k_B = sp.symbols('hbar c G M k_B', positive=True)\n\n"
            "def answer(hbar, c, G, M, k_B):\n"
            "    T_H = hbar * c**3 / (4 * sp.pi * G * M * k_B)\n"
            "    return T_H\n"
            "```\n"
        )
        result = evaluate_response(response, problem_def)
        assert result["correct"] is False

    def test_numerical_correct(self):
        """Numerical answer (Lamb shift style)."""
        problem_def = {
            "answer": 1057.8,
            "answer_template": (
                "def answer():\n    lamb_shift_MHz = ...\n    return lamb_shift_MHz\n"
            ),
        }
        response = (
            "The Lamb shift is...\n\n"
            "```python\n"
            "def answer():\n"
            "    lamb_shift_MHz = 1057.862\n"
            "    return lamb_shift_MHz\n"
            "```\n"
        )
        result = evaluate_response(response, problem_def)
        assert result["correct"] is True
        assert result["method"] == "numerical"

    def test_numerical_wrong(self):
        problem_def = {
            "answer": 1057.8,
            "answer_template": "def answer():\n    return ...\n",
        }
        response = "```python\ndef answer():\n    return 500.0\n```\n"
        result = evaluate_response(response, problem_def)
        assert result["correct"] is False

    def test_no_answer_in_yaml(self):
        result = evaluate_response("anything", {"problem": "..."})
        assert result["correct"] is None
        assert result["method"] == "no_answer"

    def test_no_code_in_response(self):
        problem_def = {
            "answer": "x + 1",
            "answer_template": "def answer():\n    return ...\n",
        }
        result = evaluate_response("Just text, no code block.", problem_def)
        assert result["correct"] is None
        assert result["method"] == "no_code"

    def test_candidate_syntax_error(self):
        problem_def = {
            "answer": 42,
            "answer_template": "def answer():\n    return ...\n",
        }
        response = "```python\ndef answer():\n    return +++\n```\n"
        result = evaluate_response(response, problem_def)
        assert result["correct"] is None
        assert "error" in result["method"]

    def test_candidate_missing_answer_func(self):
        problem_def = {
            "answer": 42,
            "answer_template": "def answer():\n    return ...\n",
        }
        response = "```python\ndef compute():\n    return 42\n```\n"
        result = evaluate_response(response, problem_def)
        assert result["correct"] is None
        assert result["method"] == "no_code"

    def test_qho_thermodynamics(self):
        """Full pipeline with QHO heat capacity (more complex expression)."""
        problem_def = {
            "answer": "C_V = k_B * (beta * hbar * omega)**2 * sp.exp(beta * hbar * omega) / (sp.exp(beta * hbar * omega) - 1)**2\n",
            "answer_template": (
                "import sympy as sp\n\n"
                "k_B, beta, hbar, omega = sp.symbols('k_B beta hbar omega', positive=True)\n\n"
                "def answer(k_B, beta, hbar, omega):\n"
                "    C_V = ...\n"
                "    return C_V\n"
            ),
        }
        # LLM writes the same expression with a different variable name
        response = (
            "```python\n"
            "import sympy as sp\n\n"
            "k_B, beta, hbar, omega = sp.symbols('k_B beta hbar omega', positive=True)\n\n"
            "def answer(k_B, beta, hbar, omega):\n"
            "    x = beta * hbar * omega\n"
            "    C_V = k_B * x**2 * sp.exp(x) / (sp.exp(x) - 1)**2\n"
            "    return C_V\n"
            "```\n"
        )
        result = evaluate_response(response, problem_def)
        assert result["correct"] is True

    def test_casimir_effect(self):
        """Casimir force with negative sign and pi factors."""
        problem_def = {
            "answer": "F_over_A = -sp.pi**2 * hbar * c / (240 * a**4)\n",
            "answer_template": (
                "import sympy as sp\n\n"
                "a, hbar, c = sp.symbols('a hbar c', positive=True)\n\n"
                "def answer(a, hbar, c):\n"
                "    F_over_A = ...\n"
                "    return F_over_A\n"
            ),
        }
        response = (
            "```python\n"
            "import sympy as sp\n\n"
            "a, hbar, c = sp.symbols('a hbar c', positive=True)\n\n"
            "def answer(a, hbar, c):\n"
            "    F_over_A = -hbar * c * sp.pi**2 / (240 * a**4)\n"
            "    return F_over_A\n"
            "```\n"
        )
        result = evaluate_response(response, problem_def)
        assert result["correct"] is True

    def test_candidate_runtime_error(self):
        """answer() raises an exception at call time."""
        problem_def = {
            "answer": 42,
            "answer_template": "def answer():\n    return ...\n",
        }
        response = "```python\ndef answer():\n    return 1/0\n```\n"
        result = evaluate_response(response, problem_def)
        assert result["correct"] is None
        assert result["method"] == "candidate_call_error"

    def test_empty_answer_string(self):
        result = evaluate_response(
            "anything", {"answer": "  \n", "answer_template": ""}
        )
        assert result["correct"] is None
        assert result["method"] == "no_answer"
