sudo apt-get install -y swi-prolog
pip install -r requirements.txt
swipl -g "pack_install(cplint), halt."
# Optional DL tools:
# swipl -g "pack_install(thea2), halt."
