```markdown
## Python and Sherlock Cheatsheet

### Updating Python

**Note:** Update methods vary by operating system.

#### Windows
1. Download the latest version from https://www.python.org/downloads/.
2. Run the installer and choose "Upgrade Now".

#### macOS
1. Download the latest version from https://www.python.org/downloads/mac-osx/.
2. Run the installer.

#### Linux (using package managers)
* **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt upgrade python3
  ```
* **Fedora/CentOS:**
  ```bash
  sudo dnf update python3
  ```
* **Arch Linux:**
  ```bash
  sudo pacman -Syu python
  ```

### Installing Sherlock
```bash
pip install sherlock
```

### Checking Sherlock Installation
```bash
pip show sherlock
```

### Using Sherlock

#### Terminal Usage
```bash
sherlock your_username
```

#### Python Usage
```python
import sherlock

# Basic usage
username = "your_username"
results = sherlock.search(username)

# Accessing results
for result in results:
  print(f"{result['platform']}: {result['url']}")

# Customizing search
platforms = ["twitter", "github"]
results = sherlock.search(username, platforms=platforms)
```

**Note:**
* Replace "your_username" with the actual username.
* Some platforms might have rate limits or require specific API keys.
* Use Sherlock ethically and responsibly.

### Additional Information
* **Verify Python version:** `python --version`
* **Check Sherlock version:** `sherlock --version`
* **Explore Sherlock documentation and examples for advanced usage.**
```
