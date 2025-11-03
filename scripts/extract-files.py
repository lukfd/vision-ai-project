import zipfile
import tarfile


with zipfile.ZipFile("data/source/Yelp-Photos.zip") as z:
    print("extracting tar file from zip file")
    z.extract("Yelp Photos/yelp_photos.tar", "/tmp")

with tarfile.open("/tmp/Yelp Photos/yelp_photos.tar") as t:
    print("extracting JSON + image data from tar file")
    # NOTE: this needs at leat 7 GB free AFTER the .zip files are downloaded
    # pics = [m for m in t.getmembers() if m.name.startswith("photos")]
    # t.extractall(members=pics, path="data")
    t.extract("photos.json", path="data/json")

with zipfile.ZipFile("data/source/Yelp-JSON.zip") as z:
    print("extracting tar file from zip file")
    z.extract("Yelp JSON/yelp_dataset.tar", "/tmp")

with tarfile.open("/tmp/Yelp JSON/yelp_dataset.tar") as t:
    print("extracting JSON from tar file")
    t.extract("yelp_academic_dataset_business.json", path="data/json")
    t.extract("yelp_academic_dataset_review.json", path="data/json")
    t.extract("yelp_academic_dataset_tip.json", path="data/json")
