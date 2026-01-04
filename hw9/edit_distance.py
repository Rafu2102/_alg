"""
Edit Distance (Levenshtein Distance)
----------------------------------
Operations allowed:
1. Insert
2. Delete
3. Replace

All operations cost = 1
"""

def edit_distance(s: str, t: str) -> int:
    """
    Calculate the minimum edit distance between strings s and t.
    """
    n, m = len(s), len(t)

    # dp[i][j] = minimum operations to convert s[:i] to t[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i  # delete i characters
    for j in range(m + 1):
        dp[0][j] = j  # insert j characters

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # delete
                dp[i][j - 1] + 1,        # insert
                dp[i - 1][j - 1] + cost  # replace or match
            )

    return dp[n][m]


def run_tests():
    """
    Run test cases to verify correctness
    """
    test_cases = [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("intention", "execution", 5),
        ("abc", "abc", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("horse", "ros", 3),
    ]

    print("Running Edit Distance Tests\n" + "-" * 35)

    for s, t, expected in test_cases:
        result = edit_distance(s, t)
        status = "PASS ✅" if result == expected else "FAIL ❌"
        print(f"s = '{s}', t = '{t}'")
        print(f"Expected: {expected}, Got: {result} -> {status}\n")


if __name__ == "__main__":
    run_tests()

    # User input test
    print("\nCustom Input Test")
    print("-" * 35)
    s = input("Enter first string: ")
    t = input("Enter second string: ")
    print(f"Edit Distance = {edit_distance(s, t)}")
