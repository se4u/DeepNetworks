# sudo port uninstall py36-jinja2 py36-html5lib py36-appnope  py36-backcall  py36-decorator  py36-parso  py36-jedi  py36-ptyprocess  py36-pexpect  py36-pickleshare  py36-setuptools  py36-pygments  py36-six  py36-wcwidth  py36-prompt_toolkit  py36-simplegeneric  py36-ipython_genutils  py36-traitlets  py36-tz  py36-dateutil  py36-jupyter_core  py36-tornado  py36-zmq  py36-jupyter_client  py36-jsonschema  py36-nbformat  py36-markupsafe  py36-webencodings  py36-bleach  py36-entrypoints  py36-mistune  py36-pandocfilters  py36-testpath  py36-nbconvert  py36-prometheus_client  py36-send2trash  py36-terminado  python36
sudo port install python36 py36-pip
sudo port select --set python python36
sudo port select --set python3 python36
sudo port select --set pip pip36
pip install --upgrade tensorflow tqdm keras jupyter matplotlib seaborn --user
