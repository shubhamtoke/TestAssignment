import sys
from main import RequirementsAnalyst


def run_error_handling_tests() -> int:
    analyst = RequirementsAnalyst()
    cases = [
        ("Empty inputs", "", ""),
        ("None inputs", None, None),
        ("Very short inputs", "Test", "Test"),
    ]

    print("=== ERROR HANDLING TESTS ===")
    for name, user_story, ac in cases:
        print(f"\n-- {name} --")
        try:
            result = analyst.analyze_user_story(user_story, ac)
            summary = result.replace("\n", " ") if isinstance(result, str) else str(result)
            print(summary[:300] + ("..." if len(summary) > 300 else ""))
            print("Status: PASS")
        except Exception as e:
            print(f"Status: FAIL - {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_error_handling_tests())
