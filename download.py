# import the necessary packages
import argparse
import requests
import os

# construct the argument parse and parse the arguments
# ex) python3 download.py -c pants_cargo -u pants_cargo_urls.txt -o pants
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--category", required=True,
        help="a name of images' category")
ap.add_argument("-u", "--urls", required=True,
        help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
        help="path to output directory of images")
args = vars(ap.parse_args())

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0


# loop the URLs
for url in rows:
    try: # try to download the image
        r = requests.get(url, timeout=60)

        # save the image to disk
        p = os.path.sep.join([args["output"], args["category"] + "{}.jpg".format(
            str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1

    # handle if any exceptions are thrown during the download process
    except:
        print("[INFO] error downloading {}...skipping".format(p))
