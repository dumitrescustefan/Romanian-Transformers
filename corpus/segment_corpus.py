import os, random
import stanza
from tqdm import tqdm
import multiprocessing

INPUT_FILE = os.path.join("merged","valid.txt")
GPU_SEGMENT_WORKERS = 4
CPU_SEGMENT_WORKERS = 6
SHARD_NUMBER = 10000
_, filename = os.path.split(INPUT_FILE)
PREFIX = filename[:filename.rindex(".")]
print("Processing file prefix ["+PREFIX+"]") # will extract the filename without the extension

# read number of lines
if not os.path.exists(os.path.join("segmented","split",PREFIX+"-000000.txt")):
    print("Reading number of lines ... ")
    with open(INPUT_FILE, "r", encoding='utf8') as f:
        line_count = sum(1 for line in f)
    split_size = int(line_count/SHARD_NUMBER)
    if split_size == 0:
        SHARD_NUMBER = 0
        split_size = line_count
    print("File has {:.2f}K lines, will split into {} files of {} lines each.".format(line_count/1000, SHARD_NUMBER, split_size))
else:
    print("Skipping reading lines, there is already a file with this prefix in the segmented/split folder.")

# create if not existing segmented-split folder
os.makedirs(os.path.join("segmented","split"), exist_ok=True)

# split if not existing initial file into SHARD_NUMBER files
print("Splitting file ... ")
if not os.path.exists(os.path.join("segmented","split",PREFIX+"-000000.txt")):
    inp = open(INPUT_FILE, "r", encoding='utf8')
    for i in tqdm(range(SHARD_NUMBER-1)):
        with open(os.path.join("segmented","split",PREFIX+"-"+str(i).zfill(6) + ".txt"), "w", encoding="utf8") as f:
            for j in range(split_size):
                f.write(inp.readline())
    with open(os.path.join("segmented","split",PREFIX+"-"+str(SHARD_NUMBER).zfill(6) + ".txt"), "w", encoding="utf8") as f:
        cnt = 0
        try:
            while True:
                line = inp.readline()
                if not line:
                    break
                f.write(line)
                cnt+=1
        except Exception as ex:
            pass

    print("Last file has "+str(cnt)+" lines, as it rounds up all leftover lines from the other files.")
else:
    print("Skipping splitting the file in shards, there is already a file with this prefix in the segmented/split folder.")


# run pool on each file and segment (first N processed get GPU, rest get CPU) - GPU_SEGMENT_WORKERS, CPU_SEGMENT_WORKERS
mypath = os.path.join("segmented", "split")
files_src = [f for f in os.listdir(mypath) if PREFIX in f and os.path.isfile(os.path.join(mypath, f))]
mypath = os.path.join("segmented")
files_dst = [f for f in os.listdir(mypath) if PREFIX in f and os.path.isfile(os.path.join(mypath, f))]
files = list(set(files_src) - set(files_dst))
print("Segmenting {} files ...".format(len(files)))

def segmenter_worker(input_filepath):
    my_id = multiprocessing.current_process()._identity[0]
    use_gpu = True if my_id <= GPU_SEGMENT_WORKERS else False
    output_filepath = os.path.join("segmented", "_tmp-"+str(my_id))
    output_filepath_final = os.path.join("segmented", input_filepath)
    input_filepath = os.path.join("segmented", "split", input_filepath)

    #print ("I'm {} and processing {}, GPU = {}".format(my_id, input_filepath, use_gpu))

    def _segment_buffer(buffer, nlp):
        # returns sentences
        if len(buffer) == 0:
            return []
        doc = nlp(buffer.strip())
        return [sentence.text for sentence in doc.sentences]

    def _write_sentences(sentences, output_file):
        if len(sentences) == 0:
            return
        for sentence in sentences:
            output_file.write(sentence + "\n")

    nlp = stanza.Pipeline('ro', processors='tokenize', package='rrt', use_gpu=use_gpu, verbose=False)

    output_file = open(output_filepath, "w", encoding="utf8")
    buffer = ""
    for line in open(input_filepath, "r", encoding="utf8"):
        stripped_line = line.strip()
        if stripped_line == "":
            _write_sentences(_segment_buffer(buffer, nlp), output_file)
            output_file.write("\n")
            continue
        buffer += stripped_line + " "
        if len(buffer) > 10000:
            _write_sentences(_segment_buffer(buffer, nlp), output_file)
            buffer = ""
    _write_sentences(_segment_buffer(buffer, nlp), output_file)
    os.rename(output_filepath, output_filepath_final)

data = files
pbar = tqdm(total=len(data), unit="files")
pool = multiprocessing.Pool(GPU_SEGMENT_WORKERS + CPU_SEGMENT_WORKERS)

def update(*a):
    pbar.update(1)

for f in data:
    pool.apply_async(segmenter_worker, args=(f,), callback=update)

pool.close()
pool.join()
pbar.close()