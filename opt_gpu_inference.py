import sys

# A sample script to test the model.
from utils.benchmark import Classify_Model


def main():
    cfg = sys.argv[1]
    weight_path = sys.argv[2]
    test = Classify_Model(cfg, weight_path)
    source = sys.argv[3]
    save_path = sys.argv[4]
    #test.inference(source, save_path)
    test.optimizedGpuInference(source, save_path)
    #test.benchmark(source, save_path)


if __name__ == '__main__':
    main()
