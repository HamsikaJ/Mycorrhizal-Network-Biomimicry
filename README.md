# Mycorrhizal-Network-Biomimicry
Machine learning analysis of mycorrhizal topologies under environmental stress / Problem Set #1

This project analyzes how fungal networks change their shape to survive in high-stress environments. We use these "biological design rules" to understand how to build better human infrastructure, like power grids.

## Repository Structure
* **data/**: Contains raw topological data (`mycorrhizal_samples.txt`) and processed features (`physical_summary.csv`).
* **notebooks/**: Contains the main analysis (`01_Biomimetic_Topology_Analysis.ipynb`) and the math script (`calculate_footprint.py`).
* **docs/**: Contains the chat history with AI (`AI_Assistance_Chat.md`) for transparency and credibility.
* **requirements.txt**: The list of software dependencies needed to run the analysis.

## How to Run the Analysis (Universal Setup)

### 1. Open the Project Folder
Open your IDE (VS Code, Antigravity AI, etc.) and ensure you have the **root project folder** open (the one containing `requirements.txt`).

### 2. Install Software
To ensure the computer finds the requirements file correctly, open your terminal (Ctrl + Shift + `) and use the command based on your current path:

* **If your terminal is in the main folder:**
  `python -m pip install -r requirements.txt`
* **If your terminal is inside the 'notebooks' folder:**
  `python -m pip install -r ../requirements.txt`

*(Note: If using a cloud notebook like Antigravity, you can simply run `%pip install -r requirements.txt` in the very first cell of the notebook).*

### 3. Open & Execute
1. Navigate to the `notebooks/` folder and open `01_Biomimetic_Topology_Analysis.ipynb`.
2. **Select Kernel:** Click **"Select Kernel"** (top right). If VS Code asks to "Create a Virtual Environment," click **Yes/OK**. 
3. **Choose Environment:** Select the **.venv** or **Python 3.12** interpreter where the requirements were installed.
4. **Run All:** Click **Kernel -> Restart & Run All**.

---

## Troubleshooting & Reproducibility
To ensure results are identical (Target: **0.80 Recall**) and to bypass common environment errors:

* **Installation Wait Times:** Packages like `xgboost` are large. If a **"Creating Environment"** progress bar appears at the bottom right of VS Code, **wait 5-10 minutes** for it to finish. Do not cancel the process.
* **Windows Error 9009:** If Python is "not found," search for **"App Execution Aliases"** in Windows settings and **disable** the toggles for Python. This ensures Windows uses your actual Python installation.
* **Automated Data:** The pipeline is "self-healing." If `physical_summary.csv` is missing, the notebook automatically triggers the `calculate_footprint.py` script to regenerate all geometric features from the raw data.
* **Fixed Seed:** We use `random_state=42` across all models to ensure consistent, reproducible results across all hardware and platforms.

If the notebook hangs or libraries fail to load, please check the following:

* **Handling Buffer/Execution Hangs:** If cells show a "loading" status for more than **5 minutes** without producing output, the kernel may have encountered a deadlock. **Interrupt and restart the kernel**, then select "Clear All Outputs" before re-running the cells.
* **Python Version Compatibility:** This project was developed using **Python 3.12/3.14**. Ensure `xgboost` and `scikit-learn` binaries are correctly compiled for your specific environment. If issues persist, a stable version (Python 3.12) is recommended.
* **Kernel Selection (VS Code):** Ensure the `.venv` created during installation is selected as the active Jupyter kernel. If cells spin indefinitely, use **"Developer: Reload Window"** to refresh the extension host.
* **Pathing Errors (Windows/PowerShell):** If you encounter `CommandNotFoundException` or Linux-style path errors (e.g., paths starting with `/C:/`), ensure you are running commands from a standard PowerShell terminal rather than a shell with mismatched pathing configurations.
* **Environment Reset:** If dependencies conflict, delete the `.venv` folder and re-run the `pip install -r requirements.txt` command to ensure a clean state.

---

## Scientific Results
The model confirms that **Max_Span_m** and **Hull_Area_m2** (geometric footprint) are the most critical predictors of environmental adaptation. This proves that fungal networks in resource-scarce (Xeric) zones prioritize expansive territorial reach over local network density (Degree).
