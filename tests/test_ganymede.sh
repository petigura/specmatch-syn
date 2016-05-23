#!/usr/bin/env bash
DEBUG="--debug"
OBS=rj76.279
umask u=rwx,g=rx,o=rx # Change the appropriate file permissions to read only
set -x # Want to keep tabs on values of variables
cd ${SM_DIR}
export SM_PROJDIR=${SM_HTML_DIR}
python code/py/scripts/restwav_batch.py -f --obs ${OBS}
python code/py/scripts/specmatch_batch.py ${OBS} --np 8 ${DEBUG}
python code/py/scripts/add_telluric.py ${OBS} --plot
python code/py/scripts/add_polish.py ${OBS} fm
python code/py/scripts/smplots_batch.py --panels --matches-chi --polish --quicklook ${OBS}
