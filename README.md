# Mycorrhizal-Network-Biomimicry
Machine learning analysis of mycorrhizal topologies under environmental stress / Problem Set #1

This project analyzes how fungal networks change their shape to survive in high-stress environments. We use these "biological design rules" to understand how to build better human infrastructure, like power grids.

## Repository Structure
* **data/**: Contains the raw data ("mycorrhizal_samples.txt") and processed data ("physical_summary.csv").
* **notebooks/**: Contains the main analysis ("01_Biomimetic_Topology_Analysis.ipynb") and the math script ("calculate_footprint.py").
* **requirements.txt**: The list of software needed to run this.

## How to Run the Analysis (Universal Setup)

### 1. Open the Project Folder
Open your IDE (VS Code, Antigravity, etc.) and ensure you have the root project folder open (the folder containing "requirements.txt").

### 2. Install Software
To ensure the computer finds the requirements file correctly:
1. **Right-click** your main project folder in the file sidebar and select **"Open in Integrated Terminal"**.
2. In the terminal, type: "python -m pip install -r requirements.txt" and press Enter.
   *(Note: If using a cloud notebook, you can run "%pip install -r requirements.txt" in the first cell).*



### 3. Open & Execute
1. Navigate to the `notebooks/` folder and open "01_Biomimetic_Topology_Analysis.ipynb".
2. **Select Kernel:** Click "Select Kernel" (top right) and choose **Python 3.12** or your preferred Python 3.x interpreter.
3. **Run All:** Click **Kernel -> Restart & Run All**.

## Troubleshooting & Reproducibility
To ensure results are identical (Target: **0.80 Recall**) and to bypass common environment errors:

* **Windows Error 9009:** If Python is "not found," search for **"App Execution Aliases"** in Windows settings and **disable** the toggles for Python. This ensures Windows uses your actual Python installation.
* **Automated Data:** The pipeline is "self-healing." If "physical_summary.csv" is missing, the notebook automatically triggers the regeneration script from the raw data.
* **Fixed Seed:** We use "random_state=42" to ensure consistent results across all hardware and platforms.

## Scientific Results
The model confirms that **Max_Span_m** and **Hull_Area_m2** are the most critical predictors of environmental adaptation, proving that networks in resource-scarce zones prioritize expansive reach over local density.
