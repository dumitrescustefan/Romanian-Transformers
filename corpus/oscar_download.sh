echo "Downloading OSCAR Corpus ..."
wget https://traces1.inria.fr/oscar/files/Compressed/ro_dedup.txt.gz

echo "Decompressing ..."
gunzip -d ro_dedup.txt.gz
mkdir -p raw/oscar/
mv ro_dedup.txt raw/oscar/ro_dedup.txt





