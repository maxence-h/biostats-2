import random

arr = [25, -4, 6, 6, 3, 7, 12, 8, 4, -87]
def twoWaySort(arr, n):
    # To store odd Numbers
    odd = []
    # To store Even Numbers
    even = []
    for i in range(n):
        # If number is even push them to even vector
        if arr[i] % 2 == 0:
            even.append(arr[i])
            # If number is odd push them to odd vector
        else:
            odd.append(arr[i])
            # Sort even array in ascending order
    even.sort()
    # Sort odd array in descending order
    odd.sort(reverse=True)
    i = 0
    # First store odd numbers to array
    for j in range(len(odd)):
        arr[i] = odd[j]
        i += 1
    # Then store even numbers to array
    for j in range(len(even)):
        arr[i] = even[j]
        i += 1


arr = [1, 3, 2, 7, 5, 4]
n = len(arr)
twoWaySort(arr, n)
for i in range(n):
    print(arr[i], end=" ")
def yes(arr):
    m = min(arr)
    for i in range(len(arr)):
        arr.remove(m)
        arr.insert(0, m)

def bubbleSort(arr):
    n = len(arr)
    swapped = False
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        if not swapped:
            return

def reverseBubbleSort(arr):
    n = len(arr)
    swapped = False
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] < arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        if not swapped:
            return

def choosePositive(len):
    list = []
    if len < 0:
        len = int(input("Choose a + int"))
    for i in range(len):
        val = random.randint(-10, 100)
        list.append(random.randint)
        if val % 2 == 0:
            bubbleSort(list)
        else:
            reverseBubbleSort(list)


choosePositive(6)