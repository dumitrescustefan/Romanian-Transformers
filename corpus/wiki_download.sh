#!/bin/bash
echo "Downloading wiki ..."
wget https://dumps.wikimedia.org/rowiki/20200220/rowiki-20200220-pages-articles.xml.bz2

echo "Decompressing ..."
mkdir -p raw/wiki
bzip2 -v -d rowiki-20200220-pages-articles.xml.bz2
mv rowiki-20200220-pages-articles.xml raw/wiki/
