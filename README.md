# EEG Semiology: A Comprehensive Toolkit for EEG Data Extraction, Visualization, and Analysis

## Overview

This repository contains a collection of Python scripts and Jupyter notebooks developed to support the research presented in the paper "[**Nocturnal motor events in epilepsy: Is there a defined physiological network?**](https://www.sciencedirect.com/science/article/abs/pii/S1388245719309174?via%3Dihub)" by Matthew Woolfe et al. The codebase is designed to analyze Electroencephalography (EEG) data, a recording of brain activity, to extract and visualize annotations, and explore the semiology of nocturnal movements in epilepsy patients through data-driven approaches.

## Contents

- `EEG_Analysis_Utility_Functions.py`: A script providing core functions for EEG data analysis, including but not limited to preprocessing, signal transformation, and statistical analysis tools.

- `EEG_To_Bipolar_Montage_Conversion.ipynb`: A notebook guiding users through the conversion of EEG data from a referential montage to a bipolar montage format.

- `EEG_Annotation_Extraction.ipynb`: A notebook outlining the process of extracting annotations from EEG datasets, with a focus on automated categorization and labeling of epileptiform activities and nocturnal movements.

- `EEG_Annotation_Visualization.ipynb`: A notebook offering tools for visualizing EEG annotations, including graphical representations of EEG signals and their corresponding annotations.

- `EEG_Semiology_Analysis.ipynb`: A notebook dedicated to the semiological analysis of EEG data, exploring the relationships between nocturnal movements, epileptiform activities, and specific cerebral networks.

These files are designed to be used independently based on your specific requirements. You can select and use any single file that suits your needs.

## Getting Started

### Installation

Clone this repository to your local machine using `git clone`, or download the ZIP file and extract it.

### Prerequisites

- Python 3.7 or later (tested with Python 3.7)
- Jupyter Notebook or JupyterLab

### Dependencies

After cloning the repository, use `cd EEGDataAnalysis` in your terminal to go to the new directory. Inside, you'll find a `requirements.txt` file. Install the required Python packages with this command:

```sh
pip install -r requirements.txt
```

### Running the Notebooks

- Navigate to the project directory and start Jupyter Notebook or JupyterLab. If you're not familiar with these tools, you can start them by running `jupyter notebook` or `jupyter lab` in your terminal.
- Open any of the .ipynb notebooks as per your requirements. The order is not mandatory; you can select any file you need to use.
- Run the cells in each notebook, following any additional instructions or comments provided within.

### Usage Notes

- Ensure that your EEG data files are in the correct format and directory as expected by the notebooks.
- Modify the utility functions or notebook parameters as needed to suit your specific dataset or analysis requirements. For example, if your dataset has a different structure, you might need to modify the data loading functions.
- Review the comments and documentation within each notebook for detailed instructions and explanations of the analysis steps.

### Contributing

Contributions to this codebase are welcome. Please open an issue or submit a pull request with your suggested changes or enhancements.

### Acknowledgments

This documentation was created with the assistance of GPT4. Special thanks to the authors of the paper "[Nocturnal motor events in epilepsy: Is there a defined physiological network?](https://www.sciencedirect.com/science/article/abs/pii/S1388245719309174?via%3Dihub)" for their invaluable research and insights.
