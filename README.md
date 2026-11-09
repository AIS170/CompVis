# COMP9517 Group Project

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:AIS170/CompVis.git
cd CompVis
```

If you already have the repository, run:

```bash
git pull origin main
```

### 2. Install Dependencies

Make sure Python and pip are installed.

Run the following command in the project directory:

```bash
pip install -r requirements.txt
```

This automatically installs all Python libraries required for the project.

### 3. Run the Dataset Notebook

Open Jupyter Notebook:

```bash
jupyter notebook
```

Open `data_load.ipynb` and run all cells in order.

The notebook will automatically download the **Beach Bird Detection Dataset** using KaggleHub.

**Note:** The dataset is downloaded only on the first run and is cached locally for future use. Each group member must run the notebook on their own computer.

### 4. Start Working

Once the dataset has been downloaded, you can begin developing your respective models.

**Important:** Always run `git pull origin main` before starting work to ensure you have the latest project files.
