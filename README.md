# Google Research Internship Project

This project comprises multiple research tasks spanning machine learning and quantum computing, addressing a wide range of topics from classical algorithms to quantum technologies.

## Project Structure

The project is organized into the following main tasks:

- **Task I**: Quantum Computing  
- **Task II**: Classical Graph Neural Network (GNN)  
- **Task III**: Open Task  
- **Task V**: Quantum Graph Neural Network  
- **Task VI**: Quantum Representation Learning  
- **Task VII**: Equivariant Quantum Neural Networks  
- **Task VIII**: Vision Transformer  
- **Task IX**: Kolmogorov–Arnold Network  
- **Task X**: Diffusion Models  
- **Task XI**: Multilayer Perceptron (MLP)

## Data Sources

### Task II – Classical Graph Neural Network (GNN)
- **Dataset**: Pythia8 Quark and Gluon Jets for Energy Flow  
- **DOI**: [10.5281/zenodo.3164691](https://doi.org/10.5281/zenodo.3164691)  
- **Download link**: [Zenodo Record](https://zenodo.org/records/3164691#.YigdGt9MHrB)  
- **File used**: `QG_jets.npz`

### Task X – Diffusion
- **Download link**: [Google Drive](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing)

## Environment Requirements

### Python
- Version: Python 3.8 or later  
- Package Manager: `pip` or `conda`

### Key Dependencies
- PyTorch  
- NumPy  
- SciPy  
- Matplotlib  
- Jupyter Notebook  
- Qiskit (for quantum computing tasks)  
- NetworkX (for graph neural network tasks)

## Installation Guide

1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Using conda
   conda create -n google-research python=3.8
   conda activate google-research

   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the required datasets:**
   - **Task II**: Download `QG_jets.npz` from Zenodo (link above)
   - **Task X**: Download dataset from Google Drive (link above)

## Usage

Each task has a dedicated Jupyter Notebook that can be run directly in Jupyter Notebook or JupyterLab.

## Notes

- Some quantum computing tasks may require additional quantum-specific libraries.
- Please ensure your system meets hardware requirements for quantum simulation.
- A GPU is recommended for training deep learning models efficiently.

## Contribution Guidelines

Contributions are welcome! To contribute, please ensure the following:
1. Your code adheres to the [PEP 8](https://peps.python.org/pep-0008/) style guide.  
2. Add meaningful comments and documentation.  
3. Update or add relevant test cases if applicable.

## License

© 2024 Google LLC. Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)

> This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for details on permissions and limitations.
