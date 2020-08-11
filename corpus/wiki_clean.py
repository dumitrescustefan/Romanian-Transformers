
from wiki_dump_reader import Cleaner, iterate
from text_cleaner import Cleaner as MyCleaner
import string, re, os, sys
from tqdm import tqdm

cleaner = Cleaner()
my_cleaner = MyCleaner()
lines = []

brk = 40000
print("Extracting text from xml ...")
for title, text in tqdm(iterate('raw/wiki/rowiki-latest-pages-articles.xml')):
    #if brk<=0:
    #    break
    #brk-=1

    text = cleaner.clean_text(text)
    cleaned_text, links = cleaner.build_links(text) # get text
    lines.extend(cleaned_text.splitlines())

print("Cleaning extracted text ...")
sys.stdout.flush()
cleaned_lines, stats = my_cleaner.process(lines, min_line_length=30, disable_pbar=False)
my_cleaner.print_stats(stats)


print("Post-cleaning extracted text ...")
forbidden_in = ["٭", "*", "†", "sp.", " f.", ".org", "oraș în", "localitate în", "comună în", "sat în", ".com", ".it", "o.o.", "px", ".jpg", ".gif", " n. ", ".bmp", "\\", "(n.", "\\left", "\\right", "(d.", "&nbsp;", "::", "[[", "//", ", un ora", "este un municipiu", "este o comun", "este un ora", "{{", "Period", "from:", "till:", "BackgroundColors", "canvas:", "color:", "width:", "align:", "fontsize:", "pos:", "File", "##", "==", "image:", "ISBN", "\\over", "\\math", "style", "border", "background", "Wikipedia", "id:", "bar:", "ImageSize", "height:", "DateFormat", "text:", "orientation:", "format:", "position:", "columns:", "id:", "value:", "legend:", "ScaleMajor", "increment:", "ScaleMinor", "increment:", "REDIRECT"]
forbidden_startswith = ["redirect", "Reședințe", "Locuri", "Sedii municipale", "Orașe", "Orase", "Actori", "Actri", "Localit", "Municipii", "Pagina", "List", "Secole", "Limbi", ":", "«",".",";","?","!","#"] + [x for x in string.punctuation]
forbidden_endswith = ["Descoperă",")","}","?)","aici",".ro","-lea",";"]
# ^word: regex
re1 = re.compile(r"^\w+:", re.UNICODE)
# \d)$ ex: Coreea, statul Koryo: Kojong (Wang Ch'ol) (rege din dinastia Wang, 1214-1259)
re2 = re.compile(r"\d\)$", re.UNICODE)
# ends with year
re3 = re.compile(r"\d+$", re.UNICODE)
# , 1920.$ or with ; or ,
re4 = re.compile(r",\s*\d+[\.,;]$", re.UNICODE)
# ends with a number Longchamps (Buenos Aires) 47.622
re5 = re.compile(r"\d$", re.UNICODE)
# starts with a number
re6 = re.compile(r"^\d+", re.UNICODE)


original_size = 0
clean_size = 0

lines = []
for line in tqdm(cleaned_lines):
    line = line.strip()
    original_size+=len(line)
    line = bytes(line, 'utf-8').decode('utf-8', 'ignore')

    if re1.search(line):
        continue
    if re2.search(line):
        continue
    if re3.search(line):
        continue
    if re4.search(line):
        continue
    if re5.search(line):
        continue
    if re6.search(line):
        continue

    ok = True
    for elem in forbidden_startswith:
        if line.startswith(elem):
            ok = False
            break
    if not ok:
        continue

    ok = True
    for elem in forbidden_endswith:
        if line.endswith(elem):
            ok = False
            break
    if not ok:
        continue

    ok = True
    for elem in forbidden_in:
        if elem in line:
            ok = False
            break
    if not ok:
        continue

    words = line.split()
    cap_words = 0
    dashes = 0
    for word in words:
        if word[0].isupper():
            cap_words += 1
        if word == "-":
            dashes += 1
    if cap_words/len(words) > 0.6:
        continue
    if dashes/len(words) > 0.2:
        continue

    #print(line)
    lines.append(line)
    clean_size += len(line)

print("Writing ...")
clean_file = os.path.join("clean","wiki.txt")
with open(clean_file,"w", encoding="utf8") as clean_f:
    for line in lines:
        clean_f.write(line+"\n")

print("\n\nDone. Original/clean size = {:.2f}KB / {:.2f}KB ( {:.2f}% left)".format(original_size/1024, clean_size/1024, 100.*clean_size/original_size))
