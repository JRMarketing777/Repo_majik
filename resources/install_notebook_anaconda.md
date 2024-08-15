## Jupyter Notebook Cheatsheet

### Jupyter Notebook Basics

#### Accessing the Notebook
* **Start Jupyter Notebook:**
  ```bash
  jupyter notebook
  ```
* **List Running Servers:**
  ```bash
  jupyter notebook list
  ```

#### Security
* **Set Password:**
  - Click "Setup a Password" on the login screen.
  - Follow prompts to enter token and new password.
* **Note:** Using a strong password is recommended for security.

## Jupyter Notebook Tutorial: Getting Started

### Understanding the Login Screen

Jupyter Notebook employs token-based authentication by default for security. This means you'll need a specific token to access your notebook server.

**How to Proceed:**

1. **Find your token:**
   Open your terminal and run `jupyter notebook list`. This command displays a list of running Jupyter Notebook servers with their respective tokens. Copy the token corresponding to your server.

2. **Enter the token:**
   Paste the copied token into the password field on the login screen and press Enter.

### Setting Up a Password (Optional)

While convenient, using a token is less secure than a password. Here's how to set up a password:

1. Click "Setup a Password" on the login screen.
2. Enter your token when prompted.
3. Create a new password and confirm it.

### Additional Tips

* **Security:** Prioritize security by using a strong password and limiting access to your Jupyter Notebook server.
* **Browser Cookies:** Ensure your browser allows cookies for the Jupyter Notebook server.
* **Firewall:** Check if your firewall is blocking access to the Jupyter Notebook server.

### Tutorial Resources

For a more in-depth guide on using Jupyter Notebook, explore these resources:

* **Dataquest:** [https://www.dataquest.io/blog/jupyter-notebook-tutorial/](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
* **DebugPoint:** [https://www.debugpoint.com/jupyter-notebook-tutorial/](https://www.debugpoint.com/jupyter-notebook-tutorial/)

These resources cover installation, basic usage, and advanced features.

Would you like to proceed with setting up a password or need further assistance with finding your token?


### Installation

#### Installing Jupyter Notebook on Linux
**Prerequisites:**
* Python 3.6+
* pip

**Installation:**
```bash
pip install notebook
```

#### Installing Anaconda (Optional)
Anaconda is a popular distribution for data science, including Jupyter Notebook.

1. **Download:** Visit [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual) and download the appropriate installer for your system.
2. **Run Installer:** Follow the on-screen instructions to install Anaconda.

#### Using Anaconda
* **Create a New Environment:**
  ```bash
  conda create -n myenv python=3.9 # Replace 'myenv' with desired name
  ```
* **Activate Environment:**
  ```bash
  conda activate myenv
  ```
* **Install Packages:**
  ```bash
  conda install package_name
  ```
* **List Installed Packages:**
  ```bash
  conda list
  ```

### Additional Tips
* **Create a Virtual Environment:** Use `python3 -m venv myenv` for Python-based environments.
* **Install Kernels:** Install additional kernels (e.g., R, Julia) for different programming languages.
* **Security:** Consider using authentication and running Jupyter Notebook on a specific port for added security.

**Note:** Replace `package_name` with the actual package you want to install. For example, `conda install numpy` to install NumPy.

**Remember:**
* Always refer to the official documentation for the most up-to-date information and advanced usage.
* Consider using a version control system (e.g., Git) to manage your Jupyter Notebook projects.
