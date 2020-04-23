import argparse
import os


def main():
    frozen = "_frozen" if args.frozen else ""

    for it in range(args.iterations):
        with open(os.path.join(args.output_path, "predict_rrt_upos{}_{}.conllu".format(frozen, it + 1)), "r", encoding="utf-8") as upos_file, \
            open(os.path.join(args.output_path, "predict_rrt_xpos{}_{}.conllu".format(frozen, it + 1)), "r", encoding="utf-8") as xpos_file, \
            open(os.path.join(args.output_path, "predict_rrt{}_{}.conllu".format(frozen, it + 1)), "w", encoding='utf-8') as all_file:

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
    parser.add_argument("output_path", type=str)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)

    args = parser.parse_args()

    main()
