def knapsack(W, n):  # (Remaining Weight, Number of items checked)
    # Base case
    if n == 0 or W == 0:
        return 0

    # Check if the result is already computed
    if dp[n - 1][W] != -1:
        return dp[n - 1][W]

    # If the item's weight is more than the available weight
    if wt[n - 1] > W:
        dp[n - 1][W] = knapsack(W, n - 1)
    else:
        # max(including the item, excluding the item)
        dp[n - 1][W] = max(val[n - 1] + knapsack(W - wt[n - 1], n - 1), knapsack(W, n - 1))

    return dp[n - 1][W]
n = int(input("Enter the number of items: "))
val = list(map(int, input(f"Enter the values of {n} items (space-separated): ").split()))
wt = list(map(int, input(f"Enter the weights of {n} items (space-separated): ").split()))
W = int(input("Enter the maximum weight capacity of the knapsack: "))

# Memoization table to store results of subproblems
dp = [[-1 for _ in range(W + 1)] for _ in range(n)]
print(f"The maximum value that can be obtained is: {knapsack(W, n)}")

