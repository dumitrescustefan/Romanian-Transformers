#!/bin/bash
echo "Downloading OPUS... "
echo "  * If you see an error running opus_get, please run 'pip install -r requirements.txt"
mkdir -p raw/opus
cd raw/opus
opus_get -s ro -p raw -q
unzip -o -q \*.zip
rm *.zip
rm INFO
rm LICENSE
rm README
cd ..
cd ..
echo "Done."