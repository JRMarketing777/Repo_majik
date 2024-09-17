## Taipy: Getting Started Cheat Sheet

This cheat sheet guides you through downloading, installing, and running Taipy, a web development framework in Python.

**Downloading the Taipy Repository:**

1. Open a terminal window (usually Ctrl+Alt+T on ChromeOS or Linux, Command + Space and type "Terminal" on macOS).
2. Use the `git clone` command to download the Taipy repository:

```bash
git clone https://github.com/Avaiga/taipy.git
```

This will create a local copy of the Taipy source code in your current directory.

**Installing Taipy Dependencies:**

1. Navigate to the downloaded Taipy directory using the `cd` command:

```bash
cd taipy
```

2. Install the required libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will download and install all the necessary packages Taipy needs to function.

**Running Taipy in VS Code (Optional):**

1. Open VS Code and create a new Python file (e.g., `my_taipy_app.py`).

2. **(Optional) Setting Up Virtual Environment:**
   - Consider creating a virtual environment to isolate project dependencies. Refer to VS Code documentation for creating virtual environments.

3. **Adding Taipy to PATH (if needed):**
   - If you want to run Taipy commands from VS Code's terminal without specifying the full path, you might need to add the Taipy installation directory to your PATH.

      - **Temporary Addition (for this session):**
         ```bash
         export PATH=$PATH:/path/to/taipy
         ```

      - **Permanent Addition (for future sessions):**
         - Refer to your VS Code settings to locate the user settings file (e.g., `.bashrc` for Bash or `.zshrc` for Zsh).

         - Add the following line to the settings file, replacing `/path/to/taipy` with the actual path:
           ```bash
           export PATH=$PATH:/path/to/taipy
           ```

         - Save the changes and restart VS Code for the new PATH to take effect.

4. In your Python file, you can now import Taipy and start developing your web application.

**Example Code:**

```python
from taipy import Taipy

app = Taipy()

@app.route("/")
async def index(request):
  return {"message": "Hello, Taipy!"}

if __name__ == "__main__":
  app.run()
```

**Explanation:**

1. We import the `Taipy` class from the `taipy` module.
2. We create an instance of the `Taipy` class (`app`).
3. We define a route handler function `index` using the `@app.route("/")` decorator.
   - This function will be called when a user visits the root URL (`/`).
4. Inside the `index` function, we return a dictionary with a message.
5. Finally, we call the `app.run()` method to start the Taipy server.

**Remember:** Adjust the path to Taipy in the `export PATH` command based on your actual installation location.
## Taipy Cheat Sheet: Creating a Radio Dashboard

### Understanding the Code

This cheat sheet provides a step-by-step guide to creating a radio dashboard using Taipy.

**Key Components:**

* **`Taipy` class:** The core class for creating Taipy applications.
* **`Gui` class:** A subclass of `Taipy` used for creating the user interface.
* **`add_select`:** Adds a dropdown menu component to the GUI.
* **`add_slider`:** Adds a slider component to the GUI.
* **`add_button`:** Adds a button component to the GUI.
* **`@app.on_change`:** Decorator for defining functions to be called when a component's value changes.
* **`@app.on_click`:** Decorator for defining functions to be called when a button is clicked.

### Code Breakdown

```python
from taipy import Taipy, Gui

app = Taipy()
gui = Gui(app)

radio_type = gui.add_select("radio_type", options=["Ham Radio", "Normal Radio"], label="Radio Type")
frequency = gui.add_slider("frequency", min_value=88, max_value=108, label="Frequency")
mode = gui.add_select("mode", options=["AM", "FM"], label="Mode")

radio_bands = gui.add_select("radio_bands", options=["20m", "40m", "80m"], label="Band")
ham_modes = gui.add_select("ham_modes", options=["SSB", "CW", "FM"], label="Mode")

up_button = gui.add_button("up_button", label="Tune Up")
down_button = gui.add_button("down_button", label="Tune Down")

@app.on_change("radio_type")
async def update_options(change):
    if change.value == "Ham Radio":
        radio_bands.options = ["20m", "40m", "80m"]
        mode.options = ["SSB", "CW", "FM"]
    else:
        radio_bands.options = []
        mode.options = ["AM", "FM"]

@app.on_click("up_button")
async def increase_frequency(event):
    frequency.value += 1

@app.on_click("down_button")
async def decrease_frequency(event):
    frequency.value -= 1

app.run()
```

### Explanation

1. **Import Taipy and Gui:** Import the necessary modules for creating Taipy applications.
2. **Create App and GUI:** Create a Taipy application and its GUI instance.
3. **Add Components:**
   - `radio_type`: A dropdown menu to select between "Ham Radio" and "Normal Radio".
   - `frequency`: A slider to adjust the frequency.
   - `mode`: A dropdown menu to select the mode (AM or FM).
   - `radio_bands`: A dropdown menu for ham radio bands (only appears when "Ham Radio" is selected).
   - `ham_modes`: A dropdown menu for ham radio modes (only appears when "Ham Radio" is selected).
   - `up_button` and `down_button`: Buttons to increase or decrease the frequency.
4. **Link Components:**
   - The `update_options` function updates the `radio_bands` and `mode` options based on the selected `radio_type`.
   - The `increase_frequency` and `decrease_frequency` functions update the `frequency` value when the respective buttons are clicked.
5. **Run the App:** Start the Taipy application to display the dashboard in a web browser.

**Customization:**
You can customize the appearance and behavior of your dashboard by:
* **Styling:** Using Taipy's styling options to change colors, fonts, and layouts.
* **Data Binding:** Connecting components to data sources for dynamic updates.
* **Integration:** Integrating with external APIs or libraries for more features.

This cheat sheet provides a solid foundation for creating radio dashboards with Taipy. Feel free to experiment and explore Taipy's capabilities to build more complex and interactive applications.

This cheat sheet provides a basic guide to getting started with Taipy. Explore the Taipy documentation and resources for in-depth web development with Taipy.
