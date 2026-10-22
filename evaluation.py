import re
from decimal import Decimal

from agent import run_agent


# ==========================================
# 1. Evaluation Dataset
# ==========================================

test_cases = [
    {
        "question": "What is the total revenue?",
        "expected": "52000",
    },
    {
        "question": "Which brand has the highest total revenue?",
        "expected": "Toyota",
    },
    {
        "question": "What is the total revenue in Quebec?",
        "expected": "16500",
    },
    {
        "question": "How many sales records are there?",
        "expected": "5",
    },
    {
        "question": "Which region has the highest total revenue?",
        "expected": "Ontario",
    },
]


# ==========================================
# 2. Normalize Answers
# ==========================================

def normalize_answer(text: str) -> str:
    """Normalize text for comparison."""

    text = str(text).lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def answer_matches(answer: str, expected: str) -> bool:
    """Compare an exact value, preferably on the final FINAL_ANSWER line.

    Deliberately conservative: arbitrary prose is not searched for substrings.
    This checks the declared answer, not the correctness of its explanation.
    """
    text = str(answer).strip()
    lines = text.splitlines()
    if lines:
        match = re.fullmatch(r"FINAL_ANSWER:\s*(.+)", lines[-1], re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    numeric = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    if re.fullmatch(numeric, expected.strip()):
        if not re.fullmatch(numeric, text):
            return False
        return Decimal(text.replace(",", "")) == Decimal(expected.replace(",", ""))
    return normalize_answer(text) == normalize_answer(expected)


# ==========================================
# 3. Evaluate Agent
# ==========================================

def evaluate():
    correct = 0

    for i, case in enumerate(test_cases, start=1):

        try:
            result = run_agent(
                case["question"] + "\nEnd with a standalone line: "
                "FINAL_ANSWER: <value>. Use only a number (no units or currency), "
                "brand, or region as the value, without a trailing period."
            )
            answer = str(result["messages"][-1].content)
            passed = answer_matches(answer, case["expected"])
        except Exception:
            # One failed case should not abort the remaining evaluation.
            answer = "Evaluation case failed unexpectedly."
            passed = False

        if passed:
            correct += 1

        print(f"\nTest {i}: {case['question']}")
        print(f"Expected: {case['expected']}")
        print(f"Actual: {answer}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

    # Calculate accuracy
    accuracy = correct / len(test_cases) * 100 if test_cases else 0.0

    print("\n========== EVALUATION ==========")
    print(f"Total questions: {len(test_cases)}")
    print(f"Correct answers: {correct}")
    print(f"Answer Accuracy: {accuracy:.1f}%")
    return accuracy


# ==========================================
# 4. Main
# ==========================================

if __name__ == "__main__":
    evaluate()
