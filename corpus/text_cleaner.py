import re, multiprocessing
from tqdm import tqdm
import numpy as np

class Cleaner():
   def __init__(self, num_threads=1): # right now, it's single threaded
       self.num_threads = min(num_threads, int(multiprocessing.cpu_count()/2))

       """
       S- ar putea să fie necesar să- l recitiţi.
       """
       self.r1 = re.compile(r"([\w]+-)[\s]([\w]+)", re.IGNORECASE)

       """
       {LL/ AAAA}
       Humalog Mix50 100 U/ ml
       """
       self.r2 = re.compile(r"([\w]+/)\s([\w]+)", re.IGNORECASE)

       """
       All unicode dashes to normal '-', see https://www.fileformat.info/info/unicode/category/Pd/list.htm
       includes bull : • \u2022
       """
       self.r3 = re.compile(r"([■\u2022\u007E\u00AD\u058A\u05BE\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u2053\u207B\u208B\u2212\u2E17\u2E3A\u2E3B\u301C\u3030\u30A0\uFE31\uFE32\uFE63\uFF0D]+)", re.UNICODE)

       """
       spaces after comma in numbers: 1, 4% -> 1,4%
       """
       self.r4 = re.compile(r"([\d]+,)\s([\d]+)", re.IGNORECASE)

       """
       soft hyphens #\u00AD
       """
       self.r5 = re.compile(r"[\u00AD]")

       """
       remove URLS
       """
       self.r6 = re.compile(r'(?:www|http)\S+|<\S+|\w+\/*>')

       """
       remove emails
       """
       self.r7 = re.compile(r'([^@]+@[^@]+\.[^@]+)')

       """
       multiple spaces
       """
       self.space = re.compile(' +')

       """
       forbiden chars that cause a lot of bad sentences
       """
       self.forbidden_chars = "ºþÈ™ÓÑÄÈÃ®ƒ"

   def process(self, lines, percent_max_numeric=0.25, percent_max_non_ascii=0.40, min_line_length=20, verbose=False, disable_pbar=True):
       skipped_because_min_length = np.array([0,0], dtype=np.uint64)
       skipped_alpha_count = np.array([0,0], dtype=np.uint64)
       skipped_because_max_numeric = np.array([0,0], dtype=np.uint64)
       skipped_because_max_non_ascii = np.array([0,0], dtype=np.uint64)
       skipped_because_forbidden_chars = np.array([0,0], dtype=np.uint64)
       total_original_length = 0
       total_clean_length = 0
       output = []
       for line in tqdm(lines, disable = disable_pbar):
           line = line.strip()

           # get stats about line
           length = len(line)
           total_original_length += length

           if length < min_line_length:
               skipped_because_min_length += np.array([1,length], dtype=np.uint64)
               continue

           line = bytes(line, 'utf-8').decode('utf-8', 'ignore') # strip not utf-8 chars

           digit_count = 0
           alpha_count = 0
           ascii_count = 0
           forbidden_char = False
           for char in line:
               if char in self.forbidden_chars:
                   forbidden_char = True
                   break
               if char.isnumeric():
                   digit_count+=1
               if char.isalpha():
                   alpha_count+=1
               if char.isascii():
                   ascii_count+=1

           # reject if forbidden char
           if forbidden_char:
               skipped_because_forbidden_chars += np.array([1,length], dtype=np.uint64)
               continue

           # reject if number of letters is too small
           if alpha_count == 0 or alpha_count / length < 0.5:
               skipped_alpha_count += np.array([1,length], dtype=np.uint64)
               if verbose:
                   print("Skipping alpha={:.3f}: [{}]".format(alpha_count / length, line))
               continue

           # reject if too many numbers
           if digit_count / alpha_count >= percent_max_numeric and digit_count > 6:
               skipped_because_max_numeric += np.array([1,length], dtype=np.uint64)
               if verbose:
                   print("Skipping digit={:.3f}: [{}]".format(digit_count / alpha_count, line))
               continue
           # reject if too many non-ascii
           if ascii_count / alpha_count < percent_max_non_ascii and length > 15:
               skipped_because_max_non_ascii += np.array([1,length], dtype=np.uint64)
               if verbose:
                   print("Skipping ascii={:.3f}: [{}]".format(digit_count / alpha_count, line))
               continue

           # clean line
           #print("\nbef: {}".format(line))
           line = self.r1.sub(r"\1\2", line)
           line = self.r2.sub(r"\1\2", line)
           line = self.r3.sub("-", line)
           line = self.r4.sub(r"\1\2", line)
           line = self.r5.sub("", line)
           line = self.r6.sub("", line)
           line = self.r7.sub("", line)

           line = line.replace("( ă)", "(ă)")
           line = line.replace("ţ", "ț")
           line = line.replace("ş", "ș")
           line = line.replace("Ţ", "Ț")
           line = line.replace("Ş", "Ș")
           line = line.replace("Ã¢", "â")

           #print("aft: {}".format(line))

           line = self.space.sub(' ', line).strip()

           # check that after processing the line is not too short
           if len(line) < min_line_length:
               skipped_because_min_length += np.array([1,length], dtype=np.uint64)
               continue

           total_clean_length += len(line)
           output.append(line+"\n")

       # pack stats
       stats = {}
       stats["skipped_because_min_length"] = skipped_because_min_length
       stats["skipped_alpha_count"] = skipped_alpha_count
       stats["skipped_because_max_numeric"] = skipped_because_max_numeric
       stats["skipped_because_max_non_ascii"] = skipped_because_max_non_ascii
       stats["skipped_because_forbidden_chars"] = skipped_because_forbidden_chars
       stats["total_original_length"] = total_original_length
       stats["total_clean_length"] = total_clean_length

       return output, stats

   def add_stats(self, a, b):
       """
       Add two stats dict that are returned by the process function.
       This is used for multiple files
       :param a: stats dict
       :param b: stats dict
       :return: stats dict
       """
       stats = {}
       stats["skipped_because_min_length"] = a["skipped_because_min_length"] + b["skipped_because_min_length"]
       stats["skipped_alpha_count"] = a["skipped_alpha_count"] + b["skipped_alpha_count"]
       stats["skipped_because_max_numeric"] = a["skipped_because_max_numeric"] + b["skipped_because_max_numeric"]
       stats["skipped_because_max_non_ascii"] = a["skipped_because_max_non_ascii"] + b["skipped_because_max_non_ascii"]
       stats["skipped_because_forbidden_chars"] = a["skipped_because_forbidden_chars"] + b["skipped_because_forbidden_chars"]
       stats["total_original_length"] = a["total_original_length"] + b["total_original_length"]
       stats["total_clean_length"] = a["total_clean_length"] + b["total_clean_length"]
       return stats

   def print_stats(self, stats):
       print("\nCleaning statistics:")
       print("Total original length (chars) = {}".format(stats["total_original_length"]))
       print("Total length after cleaning (chars) = {}".format(stats["total_clean_length"]))
       print("Percent data kept = {:.3f} %".format(100.*stats["total_clean_length"]/stats["total_original_length"]))

       print("Skipped because line length was below minimum (lines/chars): {} ".format(stats["skipped_because_min_length"]))
       print("Skipped because line had forbidden characters (lines/chars): {} ".format(stats["skipped_because_forbidden_chars"]))
       print("Skipped because alpha count was below minimum (lines/chars): {} ".format(stats["skipped_alpha_count"]))
       print("Skipped because digit count was above maximum (lines/chars): {} ".format(stats["skipped_because_max_numeric"]))
       print("Skipped because too many non-ascii characters (lines/chars): {} ".format(stats["skipped_because_max_non_ascii"]))

text = [" - ~~~~~Păstraţi acest prospect. S- ar putea să fie necesar să- l recitiţi.",
           "- Dacă aveţi orice întrebări suplimentare, adresaţi- vă medicului dumneavoastră sau farmacistului.\n",
           "{LL/ AAAA}\n",
           "MANUALUL UTILIZATORULUI\n",
           "Vezi textul manualului mai jos.\n",
           "303 Informaţii detaliate privind acest medicament sunt disponibile pe website- ul Agenţiei Europene a Medicamentului (EMEA): http: // www. emea. europa. eu /.\n",
           "304 PROSPECT:­    \n",
           "INFORMAŢII PENTRU UTILIZATOR",
           "Humalog Mix50 100 U/ ml • • •  ~~~~",
           "Τηλ: +30 210 629 4600 España Lilly S. A.",
           "Tel: + 34- 91 663 50 00 France Lilly France S. A. S.",
           "Tél: +33 - (0) 1 55 49 34 34 Ireland Eli Lilly and Company (Ireland) Limited Tel: + 353 - (0) 1 661 4377 Ísland Icepharma hf.",
           "Sími + 354 540 8000 Italia Eli Lilly Italia S. p. A.",
           "Tel: + 39 - 055 42571 Κύπρος Phadisco Ltd Τηλ: +357 22 715000 ",
           "Luxembourg/ Luxemburg Eli Lilly Benelux S. A.",
           "Tél/ Tel: + 32 - (0) 2 548 84 84 Magyarország Lilly Hungária Kft.",
           "Tel: + 36 1 328 5100 Malta Charles de Giorgio Ltd.",
           "Κύπρος Βαρνάβας Χατζηπαναγής Λτδ 7 Ανδροκλέους CY- 1060 Λευκωσία Tηλ"]

#tt = []
#for i in range(100000):
#    tt.extend(text)
#print(len(tt))
"""
c = Cleaner(1)
lines, s1 = c.process(text)
lines, s2 = c.process(text)

stats = c.add_stats(s1, s2)

c.print_stats(s1)
c.print_stats(s2)
c.print_stats(stats)
print("DONE")
"""




