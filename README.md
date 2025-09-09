# Installation
## Prerequisites:
* Conda

##  Create and activate the environment
```
conda env create -f environment.yml
conda activate trading_bot
```
**NOTE:** `environment.yml` references a `pip:` section (-r requirements.txt).
Run the create command from the repo root so pip can find the file.

## Use in Jupyter 
```
# Add the kernel (already includes ipykernel in the env)up
python -m ipykernel install --user --name trading_bot --display-name "Python(trading_bot)"
```
Select Python (trading_bot) in JupyterLab/Notebook. 