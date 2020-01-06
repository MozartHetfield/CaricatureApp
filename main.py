import filters
import sys

def main():
    input_path = r"./input_photos/" + str(sys.argv[1])
    output_path = r"./output_photos/output_" + str(sys.argv[1])
    mode = int(sys.argv[2]) #0 for normal, 1 for bw, 2 for sepia

    #TODO make feature extraction and exaggeration

    filters.create_sketch(input_path, output_path, mode)
    if (mode == 1):
        filters.black_and_white(output_path, output_path)
    elif (mode == 2):
        filters.create_sepia(output_path, output_path)

if __name__ == '__main__':
    main()