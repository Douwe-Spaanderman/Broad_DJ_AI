### Author Douwe Spaanderman - 27 May 2020 ###

# Main run file for the machine learning #
import argparse
import time


def main():

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change media encoding")
    parser.add_argument("Path", help="path to terra workspace filtered file")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-e", dest="Encoding", nargs='?', default='Extract', help="which representation of media you would like", choices=["Extract", "One-hot", "One-hot-percentage"])

    args = parser.parse_args()
    start = time.time()
    data = main()
    end = time.time()
    print('completed in {} seconds'.format(end-start))