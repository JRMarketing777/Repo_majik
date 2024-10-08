## Downloading PDFs to a USB Stick Using the Linux Command Line: A Cheat Sheet

### Key Concepts and Terminology

* **Command Line:** A text-based interface for interacting with your computer's operating system.
* **Terminal:** An application that provides a command-line interface.
* **USB Stick:** A small, portable storage device.
* **Mount:** To make a device (like a USB stick) accessible to your computer's file system.
* **Unmount:** To safely remove a device from your computer's file system.
* **`wget`:** A command-line utility for retrieving files from the internet.
* **`lsblk`:** A command-line utility for listing block devices (like USB sticks).
* **`mount`:** A command-line utility for mounting devices.
* **`umount`:** A command-line utility for unmounting devices.

### Steps

1. **Insert the USB Stick:** Plug your USB stick into a USB port on your Linux machine.
2. **Identify the USB Stick's Device Name:**
   - Open a terminal and use the `lsblk` command:
     ```bash
     lsblk
     ```
   - Look for a device with a label like "USB Flash Drive" or a name like "sdb" or "sdc".
3. **Create a Mount Point:**
   - Create a directory to serve as a mount point:
     ```bash
     mkdir ~/usb_mount
     ```
4. **Mount the USB Stick:**
   - Use the `mount` command:
     ```bash
     sudo mount /dev/<device_name> ~/usb_mount
     ```
   - Replace `<device_name>` with the actual name of your USB stick.
5. **Download the PDF:**
   - Use the `wget` command:
     ```bash
     wget -O ~/usb_mount/downloaded_pdf.pdf https://example.com/document.pdf
     ```
   - Replace `https://example.com/document.pdf` with the actual URL of the PDF.
6. **Unmount the USB Stick:**
   - Safely remove the USB stick:
     ```bash
     sudo umount ~/usb_mount
     ```

### Additional Notes

* **Permissions:** You might need `sudo` to mount and unmount the USB stick, depending on your system's configuration.
* **Device Names:** Device names can vary. Use `lsblk` to determine the correct name for your USB stick.
* **Customizations:** You can customize the download location and file name using the `wget` command's options.

By following these steps and understanding the key concepts, you can effectively download PDFs to your USB stick using the Linux command line.

## Creating Files and Installing Applications in Linux: A Cheat Sheet

### Key Concepts and Terminology

* **Command Line:** A text-based interface for interacting with your computer's operating system.
* **Terminal:** An application that provides a command-line interface.
* **File:** A collection of data stored on your computer.
* **Directory:** A container for files and other directories, often referred to as a "folder."
* **Package Manager:** A tool for installing, updating, and removing software packages on your Linux system.
* **`cd`:** A command to change directories.
* **`touch`:** A command to create new files.
* **`sudo`:** A command to run commands with elevated privileges (superuser).
* **`apt`:** A package manager for Debian-based Linux distributions (e.g., Ubuntu, Mint).
* **`dnf`:** A package manager for Fedora-based Linux distributions (e.g., Fedora).
* **`pacman`:** A package manager for Arch-based Linux distributions (e.g., Arch Linux, Manjaro).
* **`flatpak`:** A universal package manager for Linux.
* **`snap`:** A universal package manager for Linux.

### Creating Files

1. **Open a Terminal:** You can usually find this in your applications menu or by pressing `Ctrl+Alt+T`.
2. **Navigate to your project directory:**
   ```bash
   cd ~/my_project
   ```
   This command changes the current directory to the "my_project" folder in your home directory.
3. **Create a new file:**
   ```bash
   touch main.py
   ```
   This command creates a new file named "main.py" in the current directory.
   If you encounter permission issues, use `sudo`:
   ```bash
   sudo touch main.py
   ```
   This runs the `touch` command with elevated privileges, allowing you to create files in restricted directories.

### Installing Applications

**Using a Package Manager:**

```bash
# For Debian-based distributions (e.g., Ubuntu, Mint):
sudo apt install atom evince

# For Fedora-based distributions (e.g., Fedora):
sudo dnf install atom evince

# For Arch-based distributions (e.g., Arch Linux, Manjaro):
sudo pacman -S atom evince
```

**Using Flatpak:**

```bash
flatpak install flathub org.atom.Atom
```

**Using Snap:**

```bash
snap install atom
```

**Using a graphical application store:**
* Search for and install applications in your distribution's app store.

### Opening Files

Once you've created or downloaded files, you can open them using the following methods:

* **Double-click:** In your file manager (e.g., Nautilus, Dolphin), double-click on the file to open it with the associated application.
* **Command Line:** Use the `xdg-open` command:
   ```bash
   xdg-open main.py
   ```
   Replace `main.py` with the actual filename.

**Additional Notes:**
* Commands and package names might vary based on your distribution.
* Always run commands with `sudo` if you encounter permission issues.
* If you have trouble opening a file, check if the associated application is installed and configured correctly.
## Linux Commands for Software Development: A Cheat Sheet

This cheat sheet provides a quick reference for essential Linux commands commonly used in software development workflows, including commands for creating and managing directories and files using Python and VS Code.

### File and Directory Management

* **`ls`:** Lists files and directories in the current directory.
    * `ls -l`: Lists files with detailed information (permissions, owner, size, etc.).
    * `ls -a`: Lists all files, including hidden ones.
* **`cd`:** Changes the current directory.
    * `cd ..`: Moves up one directory.
    * `cd ~`: Goes to your home directory.
* **`mkdir`:** Creates a new directory.
* **`rm`:** Removes a file.
    * `rm -r`: Removes a directory and its contents recursively (use with caution!).
    * `rm -f`: Forcefully removes a file without asking for confirmation (use with caution!).
* **`mv`:** Moves or renames a file or directory.
* **`cp`:** Copies a file or directory.

### File Viewing and Editing

* **`cat`:** Displays the contents of a file.
* **`less`:** Views files page by page, ideal for larger files.
* **`head`:** Displays the first few lines of a file.
* **`tail`:** Displays the last few lines of a file.
* **`nano`:** A simple text editor suitable for basic edits.
* **`vim`:** A powerful text editor offering advanced features and customization (learning curve involved).

### Package Management

* **`apt` (Debian/Ubuntu):** Installs, updates, and removes software packages.
    * `sudo apt install <package_name>`: Installs a package.
    * `sudo apt update`: Updates package lists.
    * `sudo apt upgrade`: Upgrades installed packages.
* **`yum` (Red Hat/CentOS):** Similar functionality to `apt` for these distributions.

### Permissions and Ownership

* **`chmod`:** Changes file or directory permissions.
* **`chown`:** Changes file or directory ownership.

### User Management

* **`id`:** Displays information about the current user.
* **`sudo`:** Allows running commands with elevated privileges (requires password).
* **`useradd`:** Creates a new user.
* **`passwd`:** Changes a user's password.

### Networking

* **`ping`:** Tests network connectivity to a host.
* **`ssh`:** Securely connects to a remote server.
* **`scp`:** Securely copies files between machines.

### Process Management

* **`ps`:** Lists running processes.
* **`top`:** Shows process information in real-time (CPU, memory usage).
* **`kill`:** Terminates a running process.

### Version Control (Git)

* Assuming Git is installed:
    * `git init`: Initializes a Git repository.
    * `git clone <url>`: Clones an existing Git repository.
    * `git add <file>`: Adds a file to the staging area.
    * `git commit -m "<message>"`: Commits changes with a message.
    * `git push origin <branch_name>`: Pushes changes to a remote repository.
    * `git pull origin <branch_name>`: Pulls changes from a remote repository.

### Creating Directories and Files from Python in VS Code

1. **Open a Python file in VS Code.**
2. **Import the `os` module:**
   ```python
   import os
   ```
3. **Use the `os.mkdir()` function to create a directory:**
   ```python
   os.mkdir("new_directory")
   ```
4. **Use the `os.makedirs()` function to create nested directories:**
   ```python
   os.makedirs("directory1/subdirectory2")
   ```
5. **Use the `open()` function to create a new file:**
   ```python
   file = open("new_file.txt", "w")
   file.write("This is the content of the new file.")
   file.close()
   ```

Remember to replace `"new_directory"` and `"new_file.txt"` with your desired names.

**Additional Tips:**

* You can use the VS Code terminal to execute these commands directly.
* For more advanced file operations, explore the `pathlib` module.
* Consider using a virtual environment for your Python projects to manage dependencies effectively.

## Linux System Information Cheat Sheet

| Command | Description |
|---|---|
| `lsb_release -a` | Displays distribution information (name, version, codename). |
| `hostnamectl` | Provides hostname, domain name, and other system information. |
| `uname -a` | Shows kernel version, CPU architecture, hostname, and more. |
| `lscpu` | Displays CPU information (model, speed, cores). |
| `lspci` | Lists PCI devices (graphics, network, sound). |
| `lsusb` | Lists USB devices. |
| `df -h` | Shows disk usage information (mount point, size, used, available). |
| `free -h` | Shows memory usage (total, used, available). |
| `dpkg -l` (Debian/Ubuntu) | Lists installed packages. |
| `rpm -qa` (Red Hat/Fedora) | Lists installed packages. |
| `pacman -Q` (Arch Linux) | Lists installed packages. |
| `htop` | Interactive task manager. |
| `ifconfig` | Displays network interface information. |
| `systemctl status` | Checks service status. |

