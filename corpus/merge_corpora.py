import os, random

valid_count = 5000
print("Keeping {} lines for validation, the rest go to training.".format(valid_count))

# create ready folder
if not os.path.exists("merged"):
   os.makedirs("merged")

# list all available corpora
entries = os.listdir('clean')
entries = [x for x in entries if ".txt" in x]
print("Merging: {}".format(entries))

line_counts = []
try:
   # open files
   for file in entries:
       print("Counting lines in: {}".format(file))
       with open(os.path.join("clean", file), "r", encoding='utf8') as f:
           line_counts.append(sum(1 for line in f))
       print("\t ... read {} lines".format(line_counts[-1]))
   total_line_count = sum(line_counts)

   # open destination files
   train_f = open(os.path.join("merged", "train.txt"), "w", encoding="utf8")
   valid_f = open(os.path.join("merged", "valid.txt"), "w", encoding="utf8")

   # pick lines
   for i, file in enumerate(entries):
       target_count = int(valid_count * line_counts[i] / total_line_count)
       to_valid_lines = sorted(list(set(sorted(random.sample([x for x in range(line_counts[i])], target_count)))))
       to_valid_lines.sort()
       print("Selecting {} ({:.2f}%) validation lines from {} ...".format(len(to_valid_lines),
                                                                          100 * target_count / valid_count, file))
       print(to_valid_lines)

       cnt = -1
       pointer = 0
       match_index = to_valid_lines[pointer]
       with open(os.path.join("clean", file), "r", encoding='utf8') as f:
           for line in f:
               cnt += 1
               if cnt == match_index:  # to_valid_lines[0]: # validation
                   valid_f.write(line)
                   pointer += 1
                   if pointer >= len(to_valid_lines):
                       match_index = -1
                   else:
                       match_index = to_valid_lines[pointer]
               else:
                   train_f.write(line)

       print(cnt)
       print(pointer)  # len(to_valid_lines))



except Exception as e:
   print(str(e))

finally:
   train_f.close()
   valid_f.close()
