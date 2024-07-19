# Python File Handling 101

This guide provides an overview of basic file handling operations in Python, including searching for files, creating new files, copying directories, and examining file contents.

## Table of Contents
1. [Importing Required Modules](#importing-required-modules)
2. [Searching for Files](#searching-for-files)
3. [Creating New Files](#creating-new-files)
4. [Copying Directories](#copying-directories)
5. [Examining File Contents](#examining-file-contents)

## Importing Required Modules

```python
import os
import shutil
import glob
```

## Searching for Files

### Using os.walk()
```python
def search_files(directory, extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))

# Usage
search_files("/path/to/directory", ".txt")
```

### Using glob
```python
def search_files_glob(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))

# Usage
txt_files = search_files_glob("/path/to/directory", "*.txt")
```

## Creating New Files

### Creating a Text File
```python
def create_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

# Usage
create_file("new_file.txt", "Hello, World!")
```

### Creating a Directory
```python
def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

# Usage
create_directory("new_directory")
```

## Copying Directories

```python
def copy_directory(src, dst):
    shutil.copytree(src, dst)

# Usage
copy_directory("source_directory", "destination_directory")
```

## Examining File Contents

### Reading Entire File
```python
def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Usage
content = read_file("example.txt")
print(content)
```

### Reading File Line by Line
```python
def read_file_lines(filename):
    with open(filename, 'r') as file:
        for line in file:
            print(line.strip())

# Usage
read_file_lines("example.txt")
```

### Getting File Information
```python
def get_file_info(filename):
    return {
        "size": os.path.getsize(filename),
        "created": os.path.getctime(filename),
        "modified": os.path.getmtime(filename)
    }

# Usage
info = get_file_info("example.txt")
print(info)
```

## Error Handling

Always use try-except blocks when dealing with file operations to handle potential errors:

```python
try:
    # File operation here
except FileNotFoundError:
    print("File not found.")
except PermissionError:
    print("Permission denied.")
except Exception as e:
    print(f"An error occurred: {e}")
```

This README provides a basic overview of file handling in Python. For more advanced operations or specific use cases, refer to the official Python documentation or additional libraries designed for file management.
```

This README.md file provides a concise yet comprehensive guide to basic file handling in Python. It covers the essential operations you mentioned: searching for files, creating new ones, copying directories, and examining file contents. Users can easily copy and adapt these code snippets for their specific needs.

The guide also includes a brief mention of error handling, which is crucial when working with files. You can expand on any section or add more examples as needed for your specific use case.

Certainly! File handling is a crucial skill in Python programming. I'll teach you about file handling using the `os` module and other relevant modules. We'll cover various aspects of file operations, directory management, and path manipulations.

1. The `os` Module

The `os` module provides a way to use operating system-dependent functionality.

```python
import os

# Get current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Change directory
os.chdir('/path/to/new/directory')

# List contents of a directory
contents = os.listdir()
print(f"Directory contents: {contents}")

# Create a new directory
os.mkdir('new_directory')

# Remove a directory
os.rmdir('directory_to_remove')

# Rename a file or directory
os.rename('old_name.txt', 'new_name.txt')

# Remove a file
os.remove('file_to_remove.txt')

# Check if a path exists
if os.path.exists('file.txt'):
    print("File exists")

# Check if a path is a file
if os.path.isfile('file.txt'):
    print("It's a file")

# Check if a path is a directory
if os.path.isdir('directory'):
    print("It's a directory")
```

2. The `os.path` Module

The `os.path` module is useful for path manipulations:

```python
import os.path

# Join path components
full_path = os.path.join('directory', 'subdirectory', 'file.txt')
print(f"Full path: {full_path}")

# Get the base name of a path
base = os.path.basename('/path/to/file.txt')
print(f"Base name: {base}")  # Outputs: file.txt

# Get the directory name of a path
dir_name = os.path.dirname('/path/to/file.txt')
print(f"Directory name: {dir_name}")  # Outputs: /path/to

# Split a path into (head, tail)
head, tail = os.path.split('/path/to/file.txt')
print(f"Head: {head}, Tail: {tail}")  # Outputs: Head: /path/to, Tail: file.txt

# Get file extension
_, extension = os.path.splitext('file.txt')
print(f"Extension: {extension}")  # Outputs: .txt
```

3. The `shutil` Module

The `shutil` module offers high-level operations on files and collections of files:

```python
import shutil

# Copy a file
shutil.copy2('source.txt', 'destination.txt')

# Copy an entire directory
shutil.copytree('source_dir', 'destination_dir')

# Move a file or directory
shutil.move('source', 'destination')

# Remove an entire directory tree
shutil.rmtree('directory_to_remove')
```

4. The `glob` Module

The `glob` module is useful for finding files/directories using patterns:

```python
import glob

# Find all .txt files in current directory
txt_files = glob.glob('*.txt')
print(f"Text files: {txt_files}")

# Find all .py files in current directory and all subdirectories
py_files = glob.glob('**/*.py', recursive=True)
print(f"Python files: {py_files}")
```

5. Reading and Writing Files

Basic file operations:

```python
# Writing to a file
with open('file.txt', 'w') as f:
    f.write('Hello, World!')

# Reading from a file
with open('file.txt', 'r') as f:
    content = f.read()
    print(f"File content: {content}")

# Appending to a file
with open('file.txt', 'a') as f:
    f.write('\nAppended text')

# Reading lines from a file
with open('file.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line.strip())
```

6. Working with CSV Files

The `csv` module is useful for working with CSV files:

```python
import csv

# Writing to a CSV file
data = [['Name', 'Age'], ['Alice', 30], ['Bob', 25]]
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Reading from a CSV file
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

7. Working with JSON Files

The `json` module is used for JSON data:

```python
import json

# Writing JSON to a file
data = {'name': 'Alice', 'age': 30}
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading JSON from a file
with open('data.json', 'r') as f:
    loaded_data = json.load(f)
    print(loaded_data)

