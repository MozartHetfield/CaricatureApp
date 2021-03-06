import filters
import sys
import caricature
import faceDetection

def main():
    
    input_path = r"./input_photos/" + str(sys.argv[1])
    output_path = r"./output_photos/output_" + str(sys.argv[1])
    output_path_caricature = r"./output_photos/caricature_output_" + str(sys.argv[1])
    mode = int(sys.argv[2]) #0 for normal, 1 for bw, 2 for sepia

    caricature.caricature(input_path, output_path)
    faceDetection.detect_face_features('shape_predictor_68_face_landmarks.dat', input_path, output_path_caricature)
    filters.create_sketch(output_path, output_path, mode)
    if (mode == 1):
        filters.black_and_white(output_path, output_path)
    elif (mode == 2):
        filters.create_sepia(output_path, output_path)

if __name__ == '__main__':

    main()
