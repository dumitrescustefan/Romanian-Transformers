#!/bin/bash
echo "Downloading wiki ..."
wget https://dumps.wikimedia.org/rowiki/latest/rowiki-latest-pages-articles.xml.bz2

echo "Decompressing ..."
mkdir -p raw/wiki
bzip2 -v -d rowiki-latest-pages-articles.xml.bz2
mv rowiki-latest-pages-articles.xml raw/wiki/
