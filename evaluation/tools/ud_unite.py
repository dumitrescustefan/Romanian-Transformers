import argparse
import os


def main():
    frozen = "_frozen" if args.frozen else ""

    with open(os.path.join("output", "predict_upos{}.conllu".format(frozen)), "r", encoding="utf-8") as upos_file, \
        open(os.path.join("output", "predict_xpos{}.conllu".format(frozen)), "r", encoding="utf-8") as xpos_file, \
        open(os.path.join("output", "predict_all{}.conllu".format(frozen)), "w", encoding='utf-8') as all_file:

        for line_upos, line_xpos in zip(upos_file, xpos_file):
            if not line_upos.startswith("#") and line_upos is not "\n":
                tokens = line_upos.split("\t")
                xpos = line_xpos.split("\t")[4]

                tokens[4] = xpos

                all_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(tokens[0], tokens[1], tokens[2],
                                                                               tokens[3], tokens[4], tokens[5],
                                                                               tokens[6], tokens[7], tokens[8],
                                                                               tokens[9],))
            else:
                all_file.write(line_upos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", action="store_true")

    args = parser.parse_args()

    main()
