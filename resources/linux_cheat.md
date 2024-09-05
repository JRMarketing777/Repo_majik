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
