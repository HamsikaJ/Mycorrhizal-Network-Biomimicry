# Mycorrhizal-Network-Biomimicry
Machine learning analysis of mycorrhizal topologies under environmental stress / Problem Set #1

This project analyzes how fungal networks change their shape to survive in high-stress environments. We use these 'biological design rules' to understand how to build better human infrastructure, like power grids.

## Repository Structure
* **data/**: Contains the raw data ("mycorrhizal_samples.txt") and processed data ("physical_summary.csv").
* **notebooks/**: Contains the main analysis ("01_Biomimetic_Topology_Analysis.ipynb") and the math script ("calculate_footprint.py").
* **requirements.txt**: The list of software needed to run this.

## How to Run the Analysis
1. **Install Software:** Run "pip install -r requirements.txt" in your terminal to install the necessary tools (Pandas, XGBoost, etc.).
2. **Open the Notebook:** Go into the "notebooks/" folder and open "01_Biomimetic_Topology_Analysis.ipynb".
3. **Run All:** Click **Kernel -> Restart & Run All**.

## Results & Reproducibility
To ensure the results are identical every time you run them:
* We used a fixed "Seed" (**random_state=42**) in the Machine Learning models.
* The model successfully predicts environmental stress with a **0.80 Recall score**.
