# brain_observatory_utilities
Tools for analysis, manipulation and visualization of  Brain Observatory data availible via the AllenSDK.

Functions in this repository depend on the AllenSDK
https://github.com/AllenInstitute/AllenSDK

# Installation

Set up a dedicated conda environment:

```
conda create -n brain_observatory_utilities python=3.8 
```

Activate the new environment:

```
conda activate brain_observatory_utilities
```

Make the new environment visible in the Jupyter 
```
pip install ipykernel
python -m ipykernel install --user --name brain_observatory_utilities
```

Install brain_observatory_utilities

if you intend to edit the source code, install in developer mode:
```
git clone https://github.com/AllenInstitute/brain_observatory_utilities.git
cd brain_observatory_utilities
pip install -e .
```

# Level of Support

We are planning on occasional updating these tools and repo structure with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
