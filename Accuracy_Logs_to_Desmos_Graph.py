import csv

csv_file = 'accuracy_logs.csv'

# Ask what the user wants to extract
print("Options:")
print("1: First column with all other columns")
print("2: A specific column")
print("3: A specific pair of columns")
choice = input("Enter your choice (1/2/3): ")

# Ask if output should be on the same line
same_line = input("Do you want all output on the same line? (y/n): ").lower() == 'y'

# Open CSV
with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)  # skip header

    # Adjust column indices to be 1-based for user input
    col_index = col1 = col2 = None
    if choice == '2':
        col_index = int(input(f"Enter column number (1-{len(headers)}): ")) - 1
    elif choice == '3':
        col1 = int(input(f"Enter first column number (1-{len(headers)}): ")) - 1
        col2 = int(input(f"Enter second column number (1-{len(headers)}): ")) - 1

    output = []

    for row in reader:
        # Convert row values to floats, skip row if invalid
        try:
            numbers = [float(x) for x in row]
        except ValueError:
            continue
        
        if choice == '1':
            first = numbers[0]
            for other in numbers[1:]:
                output.append((first, other))
        
        elif choice == '2':
            output.append((numbers[col_index],))
        
        elif choice == '3':
            output.append((numbers[col1], numbers[col2]))
        
        else:
            print("Invalid choice.")
            break

# Print results
if same_line:
    print(", ".join(str(item) for item in output))  # comma-separated
else:
    for item in output:
        print(item)
