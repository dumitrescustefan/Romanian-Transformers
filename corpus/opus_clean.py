import os, ntpath
from util import *
from lxml import etree
from tqdm import tqdm
from text_cleaner import Cleaner

opus_folder = os.path.join("raw", "opus")


print("Getting list of files ...")
files = getListOfFiles(os.path.join(opus_folder))
files = [x for x in files if ".xml" in x]
print("\tI found {} total files.".format(len(files)))

# exception handling for OpenSubtitles which has many almost equivalent subtitles for the same movie
ok_files = {}
print("Filtering OpenSubtitles ...")
for file in tqdm(files, leave=False):
    # extract only the biggest file in a folder
    if "OpenSubtitles" in file:
        # extract last folder
        last_folder, _ = ntpath.split(file)

        # get all file sizes in 'last_folder'
        candidates = os.listdir(last_folder)
        if len(candidates) > 1:
            candidates_sizes = []
            for candidate in candidates:
                statinfo = os.stat(os.path.join(last_folder,candidate))
                candidates_sizes.append(statinfo.st_size)
            best_candidate = candidates_sizes.index(max(candidates_sizes))
            ff = os.path.join(last_folder,candidates[best_candidate])
            if ff not in ok_files:
                ok_files[ff]=1
        else:
            if file not in ok_files:
                ok_files[file]=1
    else:
        ok_files[file]=1

files = [ k for k in ok_files ]
print("\tFiltered OpenSubtitles and left {} total files.".format(len(files)))

print("Extracting raw texts...")
exceptions = 0
for i, file in enumerate(tqdm(files)):
    # split filename
    folder, filename = ntpath.split(file)

    # create temporary destination folder
    dest_folder = "tmp"+folder[3:]

    dest_file = os.path.join(dest_folder, filename)
    if os.path.exists(dest_file):
        continue

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # parse xml and remove all tags
    try:
        tree = etree.parse(file)
        notags = etree.tostring(tree, encoding='utf8', method='text')
        # write document
        with open(dest_file, "w", encoding="utf8") as f:
            f.write(notags.decode("utf-8"))

    except Exception as e:
        statinfo = os.stat(file)
        print("{} : size {} KB".format(e, int(statinfo.st_size)/1024))
        exceptions += 1

print("Done extracting raw texts, with {} files skipped.".format(exceptions))
print("")

print("Getting list of extracted files ...")
tmp_folder = os.path.join("tmp", "opus")
if not os.path.exists(tmp_folder):
   os.makedirs(tmp_folder)
files = getListOfFiles(os.path.join(tmp_folder))
files = [x for x in files if ".xml" in x]
print("\tI found {} total files.".format(len(files)))


print("Cleaning the texts and consolidating the corpus ...")
if not os.path.exists("clean"):
    os.makedirs("clean")
clean_file = os.path.join("clean","opus.txt")

cleaner = Cleaner()
stats = None
with open(clean_file,"w", encoding="utf8") as clean_f:
    for file in tqdm(files):
        with open(file,"r", encoding="utf8") as f:
            lines = f.readlines()

        # clean file
        clean_lines, file_stats = cleaner.process(lines)

        # add stats
        if stats == None:
            stats = file_stats
        else:
            stats = cleaner.add_stats(stats, file_stats)

        for line in clean_lines:
            clean_f.write(line)

cleaner.print_stats(stats)

import shutil
shutil.rmtree("tmp")

print("Done.")