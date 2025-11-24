#!/bin/bash

# Detect OS
OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)  # macOS
else
    NUM_CORES=$(nproc)  # Linux
fi


echo
echo
echo ------- ACU-selfadapt -------
echo
make clean
make sim -j$NUM_CORES


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
