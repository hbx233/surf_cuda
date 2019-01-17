#!/bin/bash
export DEST="./.exvim.surf_cuda"
export TOOLS="/home/hbx/.vim/tools/"
export TMP="${DEST}/_inherits"
export TARGET="${DEST}/inherits"
sh ${TOOLS}/shell/bash/update-inherits.sh
