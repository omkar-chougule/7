import random

# QuickSort function for both deterministic and randomized
def quick_sort(arr, low, high, randomized=False, comp=[0]):
    if low < high:
        pivot_index = partition(arr, low, high, randomized, comp)
        quick_sort(arr, low, pivot_index - 1, randomized, comp)
        quick_sort(arr, pivot_index + 1, high, randomized, comp)

# Partition function for both deterministic and randomized variants
def partition(arr, low, high, randomized, comp):
    comp[0] += high - low
    if randomized:
        rand_pivot = random.randint(low, high)
        arr[high], arr[rand_pivot] = arr[rand_pivot], arr[high]
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Main function to analyze QuickSort variants
def analyze_quick_sort():
    arr = list(map(int, input("Enter array elements: ").split()))
    arr_copy = arr[:]

    # Deterministic QuickSort
    det_comp = [0]
    quick_sort(arr, 0, len(arr) - 1, False, det_comp)
    print(f"Deterministic QuickSort: {arr}, Comparisons: {det_comp[0]}")

    # Randomized QuickSort
    rand_comp = [0]
    quick_sort(arr_copy, 0, len(arr_copy) - 1, True, rand_comp)
    print(f"Randomized QuickSort: {arr_copy}, Comparisons: {rand_comp[0]}")

if __name__ == "__main__":
    analyze_quick_sort()