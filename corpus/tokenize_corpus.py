import os, json, sys
from tqdm import tqdm
import multiprocessing
import logging, torch
from transformers import *
logging.basicConfig(level=logging.DEBUG)

PREFIX = "train"
LOWERCASE = True
TOKENIZED_FOLDER = "bert-50000-uncased-512"
TOKENIZER_CONFIG_FOLDER = os.path.join("..", "configs", "bert_base-vocab_50000_uncased-blocksize_512") # auto select "vocab.txt"
CPU_TOKENIZE_WORKERS = 1
BLOCK_SIZE=512

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.debug("Processing file prefix ["+PREFIX+"]")

# create if not existing tmp-tokenized folder
os.makedirs(os.path.join("tokenized", TOKENIZED_FOLDER), exist_ok=True)

# run pool on each file and tokenize (CPU_TOKENIZE WORKERS)
mypath = os.path.join("segmented")
files_src = [f.replace(".txt","") for f in os.listdir(mypath) if PREFIX in f and os.path.isfile(os.path.join(mypath, f))]
mypath = os.path.join("tokenized", TOKENIZED_FOLDER)
files_dst = [f.replace(".th","") for f in os.listdir(mypath) if PREFIX in f and os.path.isfile(os.path.join(mypath, f))]
files = list(set(files_src) - set(files_dst))
logging.debug("Tokenizing {} files ...".format(len(files)))

def split_long_line(line, tokenizer, max_len):
    full_blocks = []
    cnt = 0
    while True:
        cnt += 1
        if cnt>20: # this means we have some sort of error and we are stuck in this loop
            return [], ""
        len_line_tokenized = len(tokenizer.encode(line, add_special_tokens=False))
        #logging.debug("--------------------\nStep in split_line: {}".format(len_line_tokenized))
        #logging.debug(line)
        # remainder is less than max len
        if len_line_tokenized < max_len - 2:
            return full_blocks, line

        # search for the place to cut first part
        parts = line.split(" ")
        initial_split_point = 1 #int(len(parts) * max_len / len_line_tokenized) - 5
        i = 0
        for i in range(max(1,initial_split_point), len(parts)):
            possible_split =  " ".join(parts[:i])
            #logging.debug(possible_split)
            len_possible_split = len(tokenizer.encode(possible_split, add_special_tokens=False))
            #logging.debug("\t\t Trying to split at word {}, with len {}".format(i, len_possible_split))
            if len_possible_split > max_len - 2: # we're over the limit
                break

        if i == 0:
            return [], ""

        current_split = " ".join(parts[:i-1]).strip()
        full_blocks.append(tokenizer.encode(current_split))
        line = line[len(current_split):].strip()

def _check_ok(line, tok):
    if len(tok)>BLOCK_SIZE:
        logging.debug(">>>>> LEN ERROR:")
        logging.debug(line)
        logging.debug(len(tok))
        input("STEP")

def tokenizer_worker(input_filepath):
    logging.debug("1")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_CONFIG_FOLDER, do_lower_case=LOWERCASE)
    output_filepath = os.path.join("tokenized", TOKENIZED_FOLDER, input_filepath + ".th")
    input_filepath = os.path.join("segmented", input_filepath + ".txt")
    logging.debug("2")
    arr = []
    buffer = ""
    cntt = 0
    for line in open(input_filepath, "r", encoding="utf8"):
        #logging.debug("\n==================================================================")
        #logging.debug(len(line.split()))
        #cntt+=1
        #logging.debug("{} -> [{}]".format(cntt, line.strip()))
        #sys.stdout.flush()
        #logging.debug("PROCESSING LINE ): \n{}---\t current buffer is :\n{}\n---------------------".format(line, buffer))
        if line.strip() == "": #document level, skip
            pass

        # estimate line
        line_tokenized = tokenizer.encode(line, add_special_tokens=False)
        len_line_tokenized = len(line_tokenized)
        #logging.debug(len_line_tokenized)
        # if size is greater than a full block
        if len_line_tokenized > BLOCK_SIZE - 2:
            #logging.debug("Line is :----\n{}\n----".format(line))
            # dump existing block
            if len(buffer) > 0:
                arr.append(tokenizer.encode(buffer))
                #_check_ok(buffer.strip(), arr[-1])
                buffer = ""
            # split line in blocks and remainder
            full_blocks, remainder_line = split_long_line(line, tokenizer, BLOCK_SIZE)
            if len(full_blocks) == 0: # error checking, skip line
                #logging.debug("LINE HAS ERRORS")
                #logging.debug(line)
                #input("asd")
                continue
            # write full blocks (already tokenized, with special tokens added, all full size)
            for block in full_blocks:
                arr.append(block)
                #_check_ok("BLOCK..", arr[-1])
                #logging.debug("\nFINAL full block has {} tokens, is:\n{}".format(len(block), tokenizer.decode(block)))
                #input("next block...")
            #logging.debug("remainder: {}".format(remainder_line))
            # move to next line
            buffer = remainder_line
            continue

        # see if adding this line will overflow the buffer
        len_buffer_tokenized = len(tokenizer.encode(buffer, add_special_tokens=False))
        if len_buffer_tokenized + len_line_tokenized > BLOCK_SIZE - 2:  # overflows
            # write buffer and load this line as a new buffer
            #logging.debug("OVERFLOW - BUFFER: \n{}".format(buffer))
            #logging.debug("OVERFLOW - LINE: \n{}".format(line))
            #input("OVERFLOW???")
            arr.append(tokenizer.encode(buffer.strip()))
            #_check_ok(buffer.strip(), arr[-1])
            buffer = line.strip()
        else:  # does not overflow, add to buffer
            buffer = buffer + " " + line.strip()
    if len(buffer.strip()) > 0:
        arr.append(tokenizer.encode(buffer.strip()))

    for t in arr:
        assert len(t) <= BLOCK_SIZE, "there is an error of length {} for file {}".format(len(t), input_filepath)

    json.dump(arr, open(output_filepath, "w", encoding="utf8"))

    padded_arr = []
    fill = 0
    for inst in arr:
        fill += len(inst)
        padded_inst = inst + [0]*(BLOCK_SIZE-len(inst))
        padded_arr.append(padded_inst)
    logging.debug("Fill rate for {} is {} blocks with {:.2f}".format(input_filepath, len(arr), 100*fill/(BLOCK_SIZE*len(arr))))
    tensor = torch.tensor(padded_arr, dtype = torch.long)

    torch.save(tensor, output_filepath)

files.sort()

#tokenizer_worker(files[0])
#sys.exit(0)

data = files
pbar = tqdm(total=len(data), unit="file")
pool = multiprocessing.Pool(CPU_TOKENIZE_WORKERS)

def update(*a):
    pbar.update(1)

for f in data:
    pool.apply_async(tokenizer_worker, args=(f,), callback=update)

pool.close()
pool.join()
pbar.close()