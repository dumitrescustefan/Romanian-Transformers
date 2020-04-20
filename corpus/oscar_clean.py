from tqdm import tqdm
from text_cleaner import Cleaner
import os, sys


cleaner = Cleaner()
stats = None

total_lines = 0

if not os.path.exists("clean"):
    os.makedirs("clean")

print("Processing OSCAR corpus ... \n")
with open(os.path.join("clean","oscar.txt"), "w", encoding="utf8") as dst_f:
    with open(os.path.join("raw", "oscar", "ro_dedup.txt"), "r", encoding="utf8") as src_f:
        done = False
        while not done:
            limit = 10000
            lines = []
            while limit > 0:
                line = src_f.readline()
                if not line:
                    done = True
                    break
                lines.append(line)
                limit -= 1
            total_lines += len(lines)

            # clean batch
            clean_lines, file_stats = cleaner.process(lines)

            # add stats
            if stats == None:
                stats = file_stats
            else:
                stats = cleaner.add_stats(stats, file_stats)

            # write batch
            for line in clean_lines:
                dst_f.write(line)

            if total_lines%100000 == 0:
                print("Intermediate stats:")
                cleaner.print_stats(stats)
            print("Processed {:.2f} M of about 33.8 M lines so far ... ".format(total_lines/1000/1000))

        print("Processing done.")
