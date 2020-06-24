



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change media encoding")
    parser.add_argument("Path", help="path to terra workspace filtered file after media.py")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")
    parser.add_argument("-o", dest="Data_ordering", nargs='?', default="Occurence", help="would you like to do order the data", choices=["Occurence", "Clustered", "Nothing", "All"])
    parser.add_argument("-c", dest="Data_scaling", nargs='?', default="log2", help="how would you like to scale the data", choices=["log2", "normalized", "log10"])

    args = parser.parse_args()
    start = time.time()
    heatmap_media_matrix(data=False, Path=args.Path, Order=args.Data_ordering, save=args.Save, show=bool(args.Show), Scale=args.Data_scaling)
    end = time.time()
    print('completed in {} seconds'.format(end-start))