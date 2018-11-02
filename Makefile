run: 
    pipenv shell
    jupyter notebook
setup_dev:
    pipenv install --dev
    jupyter serverextension enable --py jupygit --sys-prefix
    jupyter nbextension install --py jupygit --sys-prefix
    jupyter nbextension enable --py jupygit --sys-prefix
    jupyter contrib nbextension install --user
