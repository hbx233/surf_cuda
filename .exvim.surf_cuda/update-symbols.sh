#!/bin/bash
export DEST="./.exvim.surf_cuda"
export TOOLS="/home/hbx/.vim/tools/"
export TMP="${DEST}/_symbols"
export TARGET="${DEST}/symbols"
sh ${TOOLS}/shell/bash/update-symbols.sh
