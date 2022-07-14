# Setup jupyter config to open md and py files as notebooks by default.
mkdir -p ${HOME}/.jupyter/labconfig
rm -f ${HOME}/.jupyter/labconfig/default_setting_overrides.json
cat << STR > ${HOME}/.jupyter/labconfig/default_setting_overrides.json
{
  "@jupyterlab/docmanager-extension:plugin": {
    "defaultViewers": {
      "markdown": "Jupytext Notebook",
      "python": "Jupytext Notebook"
    }
  }
}
STR

