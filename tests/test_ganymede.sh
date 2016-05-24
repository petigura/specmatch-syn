#!/usr/bin/env bash
DEBUG="--debug"
OBS=rj76.279
umask u=rwx,g=rx,o=rx # Change the appropriate file permissions to read only
set -x # Want to keep tabs on values of variables
cd ${SM_DIR}
export SM_PROJDIR=${SM_HTML_DIR}
#python bin/restwav_batch.py -f --obs ${OBS}
#python bin/specmatch_batch.py ${OBS} ${DEBUG}
#python bin/add_telluric.py ${OBS} --plot
#python bin/add_polish.py ${OBS} fm
python bin/smplots_batch.py --panels --matches-chi --polish --quicklook ${OBS}
