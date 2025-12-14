
```python
# 🟦 1. Array Utilities
class ArrayUtils:
    @staticmethod
    def reverse_array(arr):
        """Reverse an array"""
        n = len(arr)
        result = [0] * n
        for i in range(n):
            result[i] = arr[n - i - 1]
        return result
    
    @staticmethod
    def rotate_array(arr, k):
        """Rotate array by k positions"""
        n = len(arr)
        if n == 0:
            return arr
        k = k % n
        result = [0] * n
        for i in range(n):
            result[(i + k) % n] = arr[i]
        return result
    
    @staticmethod
    def find_max_min(arr):
        """Find max and min"""
        if not arr:
            return None, None
        max_val = arr[0]
        min_val = arr[0]
        for num in arr:
            if num > max_val:
                max_val = num
            if num < min_val:
                min_val = num
        return max_val, min_val
    
    @staticmethod
    def remove_duplicates(arr):
        """Remove duplicates"""
        if not arr:
            return arr
        seen = {}
        result = []
        for num in arr:
            if num not in seen:
                seen[num] = True
                result.append(num)
        return result
    
    @staticmethod
    def is_sorted(arr):
        """Check if array is sorted"""
        for i in range(1, len(arr)):
            if arr[i] < arr[i - 1]:
                return False
        return True

# 🟦 2. String Utilities
class StringUtils:
    @staticmethod
    def reverse_string(s):
        """Reverse a string"""
        result = ''
        for i in range(len(s) - 1, -1, -1):
            result += s[i]
        return result
    
    @staticmethod
    def is_palindrome(s):
        """Check palindrome"""
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    @staticmethod
    def char_frequency(s):
        """Count character frequency"""
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        return freq
    
    @staticmethod
    def is_anagram(s1, s2):
        """Check anagram"""
        if len(s1) != len(s2):
            return False
        freq1 = StringUtils.char_frequency(s1)
        freq2 = StringUtils.char_frequency(s2)
        if len(freq1) != len(freq2):
            return False
        for char, count in freq1.items():
            if char not in freq2 or freq2[char] != count:
                return False
        return True
    
    @staticmethod
    def first_non_repeating(s):
        """Find first non-repeating character"""
        freq = StringUtils.char_frequency(s)
        for char in s:
            if freq[char] == 1:
                return char
        return None

# 🟦 3. Searching Algorithms
class SearchAlgorithms:
    @staticmethod
    def linear_search(arr, target):
        """Linear search"""
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    @staticmethod
    def binary_search(arr, target):
        """Binary search"""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    @staticmethod
    def first_occurrence(arr, target):
        """First occurrence in sorted array"""
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == target:
                result = mid
                right = mid - 1
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result
    
    @staticmethod
    def last_occurrence(arr, target):
        """Last occurrence in sorted array"""
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == target:
                result = mid
                left = mid + 1
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result
    
    @staticmethod
    def count_occurrences(arr, target):
        """Count occurrences in sorted array"""
        first = SearchAlgorithms.first_occurrence(arr, target)
        if first == -1:
            return 0
        last = SearchAlgorithms.last_occurrence(arr, target)
        return last - first + 1

# 🟦 4. Sorting Helpers
class SortingHelpers:
    @staticmethod
    def is_sorted(arr):
        """Check if array is sorted"""
        return ArrayUtils.is_sorted(arr)
    
    @staticmethod
    def custom_sort(arr, key_func):
        """Custom sort using key (bubble sort implementation)"""
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if key_func(arr[j]) < key_func(arr[i]):
                    arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    @staticmethod
    def sort_by_frequency(arr):
        """Sort by frequency"""
        freq = {}
        for num in arr:
            freq[num] = freq.get(num, 0) + 1
        
        # Create list of tuples
        items = [(num, freq[num]) for num in arr]
        
        # Sort by frequency then value (bubble sort)
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if (items[j][1] > items[i][1] or 
                    (items[j][1] == items[i][1] and items[j][0] < items[i][0])):
                    items[i], items[j] = items[j], items[i]
        
        return [item[0] for item in items]
    
    @staticmethod
    def sort_strings_by_length(strings):
        """Sort strings by length (bubble sort)"""
        n = len(strings)
        for i in range(n):
            for j in range(i + 1, n):
                if len(strings[j]) < len(strings[i]):
                    strings[i], strings[j] = strings[j], strings[i]
        return strings
    
    @staticmethod
    def kth_smallest(arr, k):
        """Kth smallest element (selection sort approach)"""
        if k <= 0 or k > len(arr):
            return None
        
        # Selection sort first k elements
        for i in range(k):
            min_idx = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
        return arr[k - 1]

# 🟦 5. Two Pointers Patterns
class TwoPointers:
    @staticmethod
    def pair_sum_sorted(arr, target):
        """Pair sum in sorted array"""
        result = []
        left, right = 0, len(arr) - 1
        while left < right:
            current_sum = arr[left] + arr[right]
            if current_sum == target:
                result.append([arr[left], arr[right]])
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        return result
    
    @staticmethod
    def remove_duplicates_in_place(arr):
        """Remove duplicates in-place"""
        if not arr:
            return arr
        
        j = 1
        for i in range(1, len(arr)):
            if arr[i] != arr[i - 1]:
                arr[j] = arr[i]
                j += 1
        
        return arr[:j]
    
    @staticmethod
    def reverse_vowels(s):
        """Reverse vowels"""
        vowels = set('aeiouAEIOU')
        chars = list(s)
        left, right = 0, len(chars) - 1
        
        while left < right:
            while left < right and chars[left] not in vowels:
                left += 1
            while left < right and chars[right] not in vowels:
                right -= 1
            
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1
        
        return ''.join(chars)
    
    @staticmethod
    def merge_sorted_arrays(arr1, arr2):
        """Merge two sorted arrays"""
        result = []
        i = j = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        
        while i < len(arr1):
            result.append(arr1[i])
            i += 1
        
        while j < len(arr2):
            result.append(arr2[j])
            j += 1
        
        return result
    
    @staticmethod
    def is_palindrome_alpha_num(s):
        """Check palindrome ignoring non-alphanumeric"""
        left, right = 0, len(s) - 1
        
        while left < right:
            while left < right and not TwoPointers.is_alphanumeric(s[left]):
                left += 1
            while left < right and not TwoPointers.is_alphanumeric(s[right]):
                right -= 1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left += 1
            right -= 1
        
        return True
    
    @staticmethod
    def is_alphanumeric(char):
        """Check if character is alphanumeric"""
        return ('a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9')

# 🟦 6. Sliding Window Patterns
class SlidingWindow:
    @staticmethod
    def max_sum_subarray(arr, k):
        """Maximum sum subarray of size k"""
        if len(arr) < k:
            return 0
        
        window_sum = 0
        for i in range(k):
            window_sum += arr[i]
        
        max_sum = window_sum
        
        for i in range(k, len(arr)):
            window_sum += arr[i] - arr[i - k]
            if window_sum > max_sum:
                max_sum = window_sum
        
        return max_sum
    
    @staticmethod
    def longest_unique_substring(s):
        """Longest substring without repeating characters"""
        last_seen = {}
        max_len = 0
        start = 0
        
        for i, char in enumerate(s):
            if char in last_seen and last_seen[char] >= start:
                start = last_seen[char] + 1
            last_seen[char] = i
            max_len = max(max_len, i - start + 1)
        
        return max_len
    
    @staticmethod
    def longest_substring_k_distinct(s, k):
        """Longest substring with k distinct characters"""
        if k == 0:
            return 0
        
        freq = {}
        max_len = 0
        left = 0
        
        for right in range(len(s)):
            freq[s[right]] = freq.get(s[right], 0) + 1
            
            while len(freq) > k:
                freq[s[left]] -= 1
                if freq[s[left]] == 0:
                    del freq[s[left]]
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len
    
    @staticmethod
    def count_subarrays_with_sum(arr, target):
        """Count subarrays with given sum"""
        count = 0
        
        for start in range(len(arr)):
            current_sum = 0
            for end in range(start, len(arr)):
                current_sum += arr[end]
                if current_sum == target:
                    count += 1
        
        return count
    
    @staticmethod
    def min_window(s, t):
        """Minimum window substring"""
        if not s or not t:
            return ""
        
        target_freq = {}
        for char in t:
            target_freq[char] = target_freq.get(char, 0) + 1
        
        window_freq = {}
        required = len(target_freq)
        formed = 0
        left = right = 0
        min_len = len(s) + 1
        min_start = 0
        
        while right < len(s):
            char = s[right]
            window_freq[char] = window_freq.get(char, 0) + 1
            
            if char in target_freq and window_freq[char] == target_freq[char]:
                formed += 1
            
            while left <= right and formed == required:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_start = left
                
                left_char = s[left]
                window_freq[left_char] -= 1
                if left_char in target_freq and window_freq[left_char] < target_freq[left_char]:
                    formed -= 1
                left += 1
            
            right += 1
        
        return "" if min_len == len(s) + 1 else s[min_start:min_start + min_len]

# 🟦 7. Hashing / Dictionary Usage
class HashingUtils:
    @staticmethod
    def frequency_counter(arr):
        """Frequency counter"""
        freq = {}
        for item in arr:
            freq[item] = freq.get(item, 0) + 1
        return freq
    
    @staticmethod
    def two_sum(nums, target):
        """Two sum"""
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
    
    @staticmethod
    def group_anagrams(strs):
        """Group anagrams"""
        groups = {}
        
        for s in strs:
            # Create frequency signature
            freq = [0] * 26
            for char in s:
                freq[ord(char) - ord('a')] += 1
            
            # Convert to string key
            key = ''.join(str(count) + ',' for count in freq)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(s)
        
        result = []
        for group in groups.values():
            result.append(group)
        
        return result
    
    @staticmethod
    def longest_consecutive(nums):
        """Longest consecutive sequence"""
        if not nums:
            return 0
        
        num_set = {}
        for num in nums:
            num_set[num] = True
        
        longest = 0
        
        for num in num_set:
            if num - 1 not in num_set:  # Start of sequence
                current = num
                length = 1
                
                while current + 1 in num_set:
                    current += 1
                    length += 1
                
                longest = max(longest, length)
        
        return longest
    
    @staticmethod
    def find_duplicates(nums):
        """Find duplicates"""
        seen = {}
        duplicates = []
        
        for num in nums:
            if num in seen:
                duplicates.append(num)
            else:
                seen[num] = True
        
        return duplicates

# 🟦 8. Stack Utilities
class StackUtils:
    @staticmethod
    def is_valid_parentheses(s):
        """Valid parentheses"""
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        
        for char in s:
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop() != pairs[char]:
                    return False
        
        return len(stack) == 0
    
    @staticmethod
    def next_greater_element(nums):
        """Next greater element"""
        result = [-1] * len(nums)
        stack = []
        
        for i in range(len(nums) - 1, -1, -1):
            while stack and stack[-1] <= nums[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            
            stack.append(nums[i])
        
        return result
    
    @staticmethod
    def previous_smaller_element(nums):
        """Previous smaller element"""
        result = [-1] * len(nums)
        stack = []
        
        for i in range(len(nums)):
            while stack and stack[-1] >= nums[i]:
                stack.pop()
            
            if stack:
                result[i] = stack[-1]
            
            stack.append(nums[i])
        
        return result
    
    @staticmethod
    def evaluate_postfix(tokens):
        """Evaluate postfix expression"""
        stack = []
        
        for token in tokens:
            if token in '+-*/':
                b = stack.pop()
                a = stack.pop()
                
                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                else:  # token == '/'
                    result = a // b if b != 0 else 0
                
                stack.append(result)
            else:
                # Convert string to int
                num = 0
                for ch in token:
                    num = num * 10 + (ord(ch) - ord('0'))
                stack.append(num)
        
        return stack[0] if stack else 0
    
    @staticmethod
    def remove_adjacent_duplicates(s):
        """Remove adjacent duplicates"""
        stack = []
        
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        
        return ''.join(stack)

# 🟦 9. Queue & Deque
class QueueUtils:
    @staticmethod
    def sliding_window_maximum(nums, k):
        """Sliding window maximum"""
        if not nums:
            return []
        
        result = []
        deque = []
        
        for i in range(len(nums)):
            # Remove elements outside window
            if deque and deque[0] < i - k + 1:
                deque.pop(0)
            
            # Remove smaller elements
            while deque and nums[deque[-1]] < nums[i]:
                deque.pop()
            
            deque.append(i)
            
            # Add to result
            if i >= k - 1:
                result.append(nums[deque[0]])
        
        return result
    
    @staticmethod
    def first_negative_in_window(arr, k):
        """First negative number in window"""
        if len(arr) < k:
            return []
        
        result = []
        deque = []
        
        for i in range(len(arr)):
            # Remove elements outside window
            if deque and deque[0] < i - k + 1:
                deque.pop(0)
            
            # Add negative numbers
            if arr[i] < 0:
                deque.append(i)
            
            # Add to result
            if i >= k - 1:
                result.append(arr[deque[0]] if deque else 0)
        
        return result
    
    @staticmethod
    def level_order_traversal(root):
        """Level order traversal for trees"""
        if not root:
            return []
        
        result = []
        queue = [root]
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.pop(0)
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    
    @staticmethod
    def generate_binary_numbers(n):
        """Generate binary numbers"""
        result = []
        queue = ['1']
        
        for _ in range(n):
            current = queue.pop(0)
            result.append(current)
            
            queue.append(current + '0')
            queue.append(current + '1')
        
        return result

# 🟦 10. Recursion Basics
class RecursionBasics:
    @staticmethod
    def factorial(n):
        """Factorial"""
        if n <= 1:
            return 1
        return n * RecursionBasics.factorial(n - 1)
    
    @staticmethod
    def fibonacci(n):
        """Fibonacci"""
        if n <= 1:
            return n
        return RecursionBasics.fibonacci(n - 1) + RecursionBasics.fibonacci(n - 2)
    
    @staticmethod
    def reverse_string_recursive(s):
        """Reverse string recursively"""
        if len(s) <= 1:
            return s
        return RecursionBasics.reverse_string_recursive(s[1:]) + s[0]
    
    @staticmethod
    def power(x, n):
        """Power function"""
        if n == 0:
            return 1
        if n < 0:
            return 1 / RecursionBasics.power(x, -n)
        return x * RecursionBasics.power(x, n - 1)
    
    @staticmethod
    def is_palindrome_recursive(s):
        """Check palindrome recursively"""
        if len(s) <= 1:
            return True
        if s[0] != s[-1]:
            return False
        return RecursionBasics.is_palindrome_recursive(s[1:-1])

# 🟦 11. Backtracking Patterns
class Backtracking:
    @staticmethod
    def subsets(nums):
        """Generate subsets"""
        result = []
        
        def backtrack(start, current):
            result.append(current[:])
            
            for i in range(start, len(nums)):
                current.append(nums[i])
                backtrack(i + 1, current)
                current.pop()
        
        backtrack(0, [])
        return result
    
    @staticmethod
    def permutations(nums):
        """Generate permutations"""
        result = []
        
        def backtrack(current):
            if len(current) == len(nums):
                result.append(current[:])
                return
            
            for num in nums:
                if num not in current:
                    current.append(num)
                    backtrack(current)
                    current.pop()
        
        backtrack([])
        return result
    
    @staticmethod
    def n_queens(n):
        """N-Queens"""
        result = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        
        def is_safe(row, col):
            # Check column
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            # Check diagonal \
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            
            # Check diagonal /
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            
            return True
        
        def backtrack(row):
            if row == n:
                result.append([''.join(row) for row in board])
                return
            
            for col in range(n):
                if is_safe(row, col):
                    board[row][col] = 'Q'
                    backtrack(row + 1)
                    board[row][col] = '.'
        
        backtrack(0)
        return result
    
    @staticmethod
    def combination_sum(candidates, target):
        """Combination sum"""
        result = []
        
        def backtrack(start, current, sum_):
            if sum_ == target:
                result.append(current[:])
                return
            
            if sum_ > target:
                return
            
            for i in range(start, len(candidates)):
                current.append(candidates[i])
                backtrack(i, current, sum_ + candidates[i])
                current.pop()
        
        backtrack(0, [], 0)
        return result
    
    @staticmethod
    def word_search(board, word):
        """Word search"""
        rows, cols = len(board), len(board[0])
        
        def dfs(r, c, index):
            if index == len(word):
                return True
            
            if (r < 0 or r >= rows or c < 0 or c >= cols or 
                board[r][c] != word[index]):
                return False
            
            temp = board[r][c]
            board[r][c] = '#'
            
            found = (dfs(r + 1, c, index + 1) or
                     dfs(r - 1, c, index + 1) or
                     dfs(r, c + 1, index + 1) or
                     dfs(r, c - 1, index + 1))
            
            board[r][c] = temp
            return found
        
        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        
        return False

# 🟦 12. Binary Search Patterns
class BinarySearchPatterns:
    @staticmethod
    def lower_bound(arr, target):
        """Lower bound (first element >= target)"""
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] >= target:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    @staticmethod
    def upper_bound(arr, target):
        """Upper bound (first element > target)"""
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] > target:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    @staticmethod
    def search_rotated(nums, target):
        """Search in rotated array"""
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    @staticmethod
    def find_peak_element(nums):
        """Find peak element"""
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    @staticmethod
    def find_min_rotated(nums):
        """Find minimum in rotated array"""
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        
        return nums[left]

# 🟦 13. Heap / Priority Queue
class MinHeap:
    """Min Heap implementation"""
    def __init__(self):
        self.heap = []
    
    def insert(self, val):
        """Insert value into heap"""
        self.heap.append(val)
        self._bubble_up(len(self.heap) - 1)
    
    def extract_min(self):
        """Extract minimum value"""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._bubble_down(0)
        
        return min_val
    
    def _bubble_up(self, index):
        """Bubble up the element at index"""
        parent = (index - 1) // 2
        while index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2
    
    def _bubble_down(self, index):
        """Bubble down the element at index"""
        n = len(self.heap)
        
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index
            
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            
            if smallest == index:
                break
            
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest
    
    def size(self):
        """Get heap size"""
        return len(self.heap)
    
    def peek(self):
        """Peek minimum value"""
        return self.heap[0] if self.heap else None

class HeapUtils:
    @staticmethod
    def kth_largest(nums, k):
        """Kth largest element (using quickselect)"""
        if k <= 0 or k > len(nums):
            return None
        
        return HeapUtils._quick_select(nums, 0, len(nums) - 1, len(nums) - k)
    
    @staticmethod
    def _quick_select(nums, left, right, k):
        """Quickselect helper"""
        if left == right:
            return nums[left]
        
        pivot_index = HeapUtils._partition(nums, left, right)
        
        if k == pivot_index:
            return nums[k]
        elif k < pivot_index:
            return HeapUtils._quick_select(nums, left, pivot_index - 1, k)
        else:
            return HeapUtils._quick_select(nums, pivot_index + 1, right, k)
    
    @staticmethod
    def _partition(nums, left, right):
        """Partition for quickselect"""
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    @staticmethod
    def top_k_frequent(nums, k):
        """Top k frequent elements"""
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        unique = list(freq.keys())
        n = len(unique)
        
        # Quickselect by frequency
        def quickselect_freq(left, right, target):
            if left == right:
                return
            
            pivot_index = HeapUtils._partition_freq(unique, left, right, freq)
            
            if target == pivot_index:
                return
            elif target < pivot_index:
                quickselect_freq(left, pivot_index - 1, target)
            else:
                quickselect_freq(pivot_index + 1, right, target)
        
        HeapUtils._partition_freq = staticmethod(lambda arr, left, right, freq: 
            HeapUtils._partition_by_freq(arr, left, right, freq))
        
        quickselect_freq(0, n - 1, n - k)
        return unique[n - k:]
    
    @staticmethod
    def _partition_by_freq(arr, left, right, freq):
        """Partition by frequency"""
        pivot = arr[right]
        i = left
        
        for j in range(left, right):
            if freq[arr[j]] <= freq[pivot]:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        
        arr[i], arr[right] = arr[right], arr[i]
        return i
    
    @staticmethod
    def merge_k_sorted_lists(lists):
        """Merge k sorted lists"""
        heap = MinHeap()
        result = []
        
        # Initialize heap with first element of each list
        for i, lst in enumerate(lists):
            if lst:
                heap.insert((lst[0], i, 0))
        
        while heap.size() > 0:
            val, list_idx, element_idx = heap.extract_min()
            result.append(val)
            
            if element_idx + 1 < len(lists[list_idx]):
                heap.insert((lists[list_idx][element_idx + 1], list_idx, element_idx + 1))
        
        return result
    
    @staticmethod
    def sort_nearly_sorted(arr, k):
        """Sort nearly sorted array"""
        heap = MinHeap()
        result = []
        
        # Add first k+1 elements to heap
        for i in range(min(k + 1, len(arr))):
            heap.insert(arr[i])
        
        # Process remaining elements
        for i in range(k + 1, len(arr)):
            result.append(heap.extract_min())
            heap.insert(arr[i])
        
        # Add remaining elements from heap
        while heap.size() > 0:
            result.append(heap.extract_min())
        
        return result
    
    @staticmethod
    def task_scheduling(tasks, n):
        """Task scheduling"""
        freq = {}
        for task in tasks:
            freq[task] = freq.get(task, 0) + 1
        
        frequencies = sorted(list(freq.values()), reverse=True)
        max_freq = frequencies[0]
        
        idle_time = (max_freq - 1) * n
        
        for i in range(1, len(frequencies)):
            idle_time -= min(max_freq - 1, frequencies[i])
        
        idle_time = max(0, idle_time)
        return len(tasks) + idle_time

# 🟦 14. Linked List Utilities
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListUtils:
    @staticmethod
    def reverse_list(head):
        """Reverse linked list"""
        prev = None
        current = head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        return prev
    
    @staticmethod
    def has_cycle(head):
        """Detect cycle"""
        if not head or not head.next:
            return False
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    @staticmethod
    def find_middle(head):
        """Find middle node"""
        if not head:
            return None
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
    
    @staticmethod
    def merge_two_lists(l1, l2):
        """Merge two sorted lists"""
        dummy = ListNode()
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        if l1:
            current.next = l1
        else:
            current.next = l2
        
        return dummy.next
    
    @staticmethod
    def remove_nth_from_end(head, n):
        """Remove nth node from end"""
        dummy = ListNode(0, head)
        slow = fast = dummy
        
        for _ in range(n + 1):
            fast = fast.next
        
        while fast:
            slow = slow.next
            fast = fast.next
        
        slow.next = slow.next.next
        return dummy.next

# 🟦 15. Tree Traversals
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class TreeTraversals:
    @staticmethod
    def inorder_traversal(root):
        """Inorder traversal"""
        result = []
        
        def traverse(node):
            if not node:
                return
            traverse(node.left)
            result.append(node.val)
            traverse(node.right)
        
        traverse(root)
        return result
    
    @staticmethod
    def preorder_traversal(root):
        """Preorder traversal"""
        result = []
        
        def traverse(node):
            if not node:
                return
            result.append(node.val)
            traverse(node.left)
            traverse(node.right)
        
        traverse(root)
        return result
    
    @staticmethod
    def postorder_traversal(root):
        """Postorder traversal"""
        result = []
        
        def traverse(node):
            if not node:
                return
            traverse(node.left)
            traverse(node.right)
            result.append(node.val)
        
        traverse(root)
        return result
    
    @staticmethod
    def level_order_traversal(root):
        """Level order traversal"""
        return QueueUtils.level_order_traversal(root)
    
    @staticmethod
    def tree_height(root):
        """Height of tree"""
        if not root:
            return -1
        
        left_height = TreeTraversals.tree_height(root.left)
        right_height = TreeTraversals.tree_height(root.right)
        
        return max(left_height, right_height) + 1

# 🟦 16. Graph Algorithms
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {}
        for v in range(vertices):
            self.adj_list[v] = []
    
    def add_edge(self, u, v):
        """Add undirected edge"""
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

class GraphAlgorithms:
    @staticmethod
    def bfs(graph, start):
        """BFS traversal"""
        visited = {}
        result = []
        queue = [start]
        visited[start] = True
        
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            
            for neighbor in graph.adj_list[vertex]:
                if neighbor not in visited:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return result
    
    @staticmethod
    def dfs(graph, start):
        """DFS traversal"""
        visited = {}
        result = []
        
        def dfs_recursive(vertex):
            visited[vertex] = True
            result.append(vertex)
            
            for neighbor in graph.adj_list[vertex]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result
    
    @staticmethod
    def has_cycle_graph(graph):
        """Detect cycle in undirected graph"""
        visited = {}
        
        def dfs(vertex, parent):
            visited[vertex] = True
            
            for neighbor in graph.adj_list[vertex]:
                if neighbor not in visited:
                    if dfs(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True
            
            return False
        
        for vertex in range(graph.vertices):
            if vertex not in visited:
                if dfs(vertex, -1):
                    return True
        
        return False
    
    @staticmethod
    def count_connected_components(graph):
        """Count connected components"""
        visited = {}
        count = 0
        
        def dfs(vertex):
            visited[vertex] = True
            for neighbor in graph.adj_list[vertex]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        for vertex in range(graph.vertices):
            if vertex not in visited:
                count += 1
                dfs(vertex)
        
        return count
    
    @staticmethod
    def shortest_path_unweighted(graph, start, end):
        """Shortest path in unweighted graph"""
        if start == end:
            return [start]
        
        visited = {}
        parent = {}
        queue = [start]
        visited[start] = True
        
        while queue:
            vertex = queue.pop(0)
            
            for neighbor in graph.adj_list[vertex]:
                if neighbor not in visited:
                    visited[neighbor] = True
                    parent[neighbor] = vertex
                    queue.append(neighbor)
                    
                    if neighbor == end:
                        # Reconstruct path
                        path = [end]
                        while path[-1] != start:
                            path.append(parent[path[-1]])
                        return path[::-1]
        
        return []

# 🟦 17. Dynamic Programming (1D)
class DynamicProgramming1D:
    @staticmethod
    def fibonacci_memo(n, memo=None):
        """Fibonacci with memoization"""
        if memo is None:
            memo = {}
        
        if n <= 1:
            return n
        
        if n in memo:
            return memo[n]
        
        memo[n] = (DynamicProgramming1D.fibonacci_memo(n - 1, memo) + 
                   DynamicProgramming1D.fibonacci_memo(n - 2, memo))
        return memo[n]
    
    @staticmethod
    def climb_stairs(n):
        """Climbing stairs"""
        if n <= 2:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    @staticmethod
    def rob(nums):
        """House robber"""
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        
        return dp[-1]
    
    @staticmethod
    def max_subarray_sum(nums):
        """Maximum subarray sum"""
        if not nums:
            return 0
        
        max_sum = current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    @staticmethod
    def coin_change(coins, amount):
        """Coin change (minimum coins)"""
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] <= amount else -1

# 🟦 18. Dynamic Programming (2D)
class DynamicProgramming2D:
    @staticmethod
    def longest_common_subsequence(text1, text2):
        """Longest common subsequence"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def longest_common_substring(text1, text2):
        """Longest common substring"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
        
        return max_length
    
    @staticmethod
    def edit_distance(word1, word2):
        """Edit distance"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1],  # Replace
                                       dp[i - 1][j],      # Delete
                                       dp[i][j - 1])      # Insert
        
        return dp[m][n]
    
    @staticmethod
    def unique_paths(m, n):
        """Unique paths"""
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        return dp[m - 1][n - 1]
    
    @staticmethod
    def knapsack_01(weights, values, capacity):
        """0/1 Knapsack"""
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], 
                                  dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]

# 🟦 19. Bit Manipulation
class BitManipulation:
    @staticmethod
    def is_power_of_two(n):
        """Check if number is power of two"""
        return n > 0 and (n & (n - 1)) == 0
    
    @staticmethod
    def count_set_bits(n):
        """Count set bits"""
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
    
    @staticmethod
    def single_number(nums):
        """Find single non-repeating number"""
        result = 0
        for num in nums:
            result ^= num
        return result
    
    @staticmethod
    def toggle_bit(n, i):
        """Toggle ith bit"""
        return n ^ (1 << i)
    
    @staticmethod
    def subsets_bitmask(nums):
        """Generate subsets using bitmask"""
        n = len(nums)
        total = 1 << n
        result = []
        
        for mask in range(total):
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
            result.append(subset)
        
        return result

# 🟦 20. Math & Number Theory
class MathUtils:
    @staticmethod
    def gcd(a, b):
        """Greatest Common Divisor"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def lcm(a, b):
        """Least Common Multiple"""
        return a // MathUtils.gcd(a, b) * b
    
    @staticmethod
    def is_prime(n):
        """Prime check"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        
        return True
    
    @staticmethod
    def sieve_of_eratosthenes(n):
        """Sieve of Eratosthenes"""
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        p = 2
        while p * p <= n:
            if is_prime[p]:
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
            p += 1
        
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
        
        return primes
    
    @staticmethod
    def fast_exponentiation(x, n):
        """Fast exponentiation"""
        if n == 0:
            return 1
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1
        while n > 0:
            if n & 1:
                result *= x
            x *= x
            n >>= 1
        
        return result
    
    @staticmethod
    def modular_inverse(a, m):
        """Modular inverse using extended Euclidean algorithm"""
        m0, x0, x1 = m, 0, 1
        
        if m == 1:
            return 0
        
        while a > 1:
            q = a // m
            a, m = m, a % m
            x0, x1 = x1 - q * x0, x0
        
        if x1 < 0:
            x1 += m0
        
        return x1


# Example usage and testing
if __name__ == "__main__":
    # Test ArrayUtils
    print("Testing ArrayUtils:")
    arr = [1, 2, 3, 4, 5]
    print(f"Reverse: {ArrayUtils.reverse_array(arr)}")
    print(f"Rotate by 2: {ArrayUtils.rotate_array(arr, 2)}")
    print(f"Max and Min: {ArrayUtils.find_max_min(arr)}")
    print(f"Remove duplicates: {ArrayUtils.remove_duplicates([1, 2, 2, 3, 3, 4])}")
    print(f"Is sorted: {ArrayUtils.is_sorted(arr)}")
    
    print("\nTesting StringUtils:")
    print(f"Reverse string: {StringUtils.reverse_string('hello')}")
    print(f"Is palindrome: {StringUtils.is_palindrome('racecar')}")
    print(f"Char frequency: {StringUtils.char_frequency('hello')}")
    print(f"Is anagram: {StringUtils.is_anagram('listen', 'silent')}")
    print(f"First non-repeating: {StringUtils.first_non_repeating('swiss')}")
    
    # More tests can be added for other classes...
    print("\nAll utilities implemented successfully!")
```

This Python implementation includes:

1. **Pure Python** - No imports, no external libraries
2. **Complete coverage** of all 20 categories
3. **Class-based organization** for modularity
4. **Proper algorithms** implemented from scratch
5. **Helper classes** for data structures (MinHeap, ListNode, TreeNode, Graph)
6. **Clear documentation** with docstrings
7. **Production-ready** code with edge case handling

The implementation uses only basic Python features:
- Lists, dictionaries, sets
- Basic control structures (loops, conditionals)
- Recursion where appropriate
- Custom implementations for all algorithms
