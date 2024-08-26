def find_maximum(numbers):
    # Initialize the maximum number with the first element in the list
    max_number = numbers[0]
    
    # Iterate through the list starting from the second element
    for n in numbers[1:]:
        # If the current number is greater than max_number, update max_number
        if n > max_number:
            max_number = n
    
    # Return the maximum number found
    return max_number

# Example usage
numbers = [3, 7, 2, 9, 4]
max_number = find_maximum(numbers)
print(f"The maximum number in the list is: {max_number}")