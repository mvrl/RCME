"""
Writes the training and validation data to folder structure organized by taxonomic hierarchy.
"""
import sys
sys.path.append("/projects/bdbl/ssastry/bioclip/src")
import argparse
import collections
import csv
import json
import logging
import multiprocessing
import os
import tarfile
import threading
import tqdm
from PIL import Image, ImageFile

from imageomics import disk_reproduce, eol_reproduce, evobio10m_reproduce, naming_reproduce, wds

########
# CONFIG
########

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
rootlogger = logging.getLogger("root")

Image.MAX_IMAGE_PIXELS = 30_000**2  # 30_000 pixels per side
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Global variables for CSV writing
csv_lock = threading.Lock()
species_counter = 0
species_folder_map = {}  # Maps species to folder names

########
# SHARED
########


def load_img(file):
    img = Image.open(file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img.resize(resize_size, resample=Image.BICUBIC)


def load_blacklists():
    image_blacklist = set()
    species_blacklist = set()

    with open(disk_reproduce.seen_in_training_json) as fd:
        for scientific, images in json.load(fd).items():
            image_blacklist |= set(os.path.basename(img) for img in images)

    with open(disk_reproduce.unseen_in_training_json) as fd:
        for scientific, images in json.load(fd).items():
            image_blacklist |= set(os.path.basename(img) for img in images)
            species_blacklist.add(scientific)

    return image_blacklist, species_blacklist


def create_taxonomic_folder_name(taxon):
    """Create folder name from taxonomic hierarchy with incrementing counter."""
    global species_counter, species_folder_map
    
    if not hasattr(taxon, 'tagged') or not taxon.tagged:
        return None
    
    # Extract taxonomic levels from tagged data
    # Assuming tagged is a list of (level, value) tuples
    # We need to get all levels from kingdom to species
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    folder_parts = []
    
    for level, value in taxon.tagged:
        if level.lower() in taxonomic_levels:
            folder_parts.append(f"{value}")
    
    # Check if we have all required levels
    if len(folder_parts) < 7:  # Need all 7 levels
        return None
    
    # Create species key for mapping
    species_key = '_'.join(folder_parts)
    
    # Thread-safe access to species counter and map
    with csv_lock:
        # Check if we've seen this species before
        if species_key in species_folder_map:
            return species_folder_map[species_key]
        
        # This is a new species, increment counter and create folder name
        species_counter += 1
        folder_name = f"{species_counter:06d}_{species_key}"
        species_folder_map[species_key] = folder_name
        
        return folder_name


def get_taxonomic_info(taxon):
    """Extract taxonomic information for CSV."""
    if not hasattr(taxon, 'tagged') or not taxon.tagged:
        return None, None, None, None, None, None, None
    
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxon_dict = {}
    
    for level, value in taxon.tagged:
        if level.lower() in taxonomic_levels:
            taxon_dict[level.lower()] = value
    
    return (
        taxon_dict.get('kingdom', ''),
        taxon_dict.get('phylum', ''),
        taxon_dict.get('class', ''),
        taxon_dict.get('order', ''),
        taxon_dict.get('family', ''),
        taxon_dict.get('genus', ''),
        taxon_dict.get('species', '')
    )


def save_image_to_folder(img, global_id, folder_name, source_dataset, taxon, common):
    """Save image to appropriate folder and write to CSV."""
    # Create folder path
    folder_path = os.path.join(outdir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Save image
    image_filename = f"{global_id}.jpg"
    image_path = os.path.join(folder_path, image_filename)
    img.save(image_path, "JPEG", quality=95)
    
    # Get taxonomic information
    kingdom, phylum, class_name, order, family, genus, species = get_taxonomic_info(taxon)
    
    # Write to CSV (thread-safe)
    csv_path = os.path.join(outdir, "image_metadata.csv")
    with csv_lock:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                global_id,
                folder_name,
                image_filename,
                source_dataset,
                os.path.join(folder_name, image_filename),
                taxon.scientific,
                common,
                kingdom,
                phylum,
                class_name,
                order,
                family,
                genus,
                species
            ])


def has_complete_taxonomy(taxon):
    """Check if taxon has all required taxonomic levels."""
    if not hasattr(taxon, 'tagged') or not taxon.tagged:
        return False
    
    required_levels = {'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'}
    found_levels = set()
    
    for level, value in taxon.tagged:
        if level.lower() in required_levels and value:
            found_levels.add(level.lower())
    
    return len(found_levels) == len(required_levels)


######################
# Encyclopedia of Life
######################


def copy_eol_from_tar(imgset_path):
    logger = logging.getLogger(f"p{os.getpid()}")

    db = evobio10m_reproduce.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, content_id, page_id FROM eol;"
    eol_ids_lookup = {
        evobio10m_id: (content_id, page_id)
        for evobio10m_id, content_id, page_id in db.execute(select_stmt).fetchall()
    }
    db.close()

    name_lookup = naming_reproduce.load_name_lookup(disk_reproduce.eol_name_lookup_json, keytype=int)
    print(f"Loaded {len(name_lookup)} EOL names.")
    # r|gz indicates reading from a gzipped file, streaming only
    with tarfile.open(imgset_path, "r|gz") as tar:
        for i, member in tqdm.tqdm(enumerate(tar)):
            eol_img = eol_reproduce.ImageFilename.from_filename(member.name)
            if eol_img.raw in image_blacklist:
                continue
            
            # Match on treeoflife_id filename
            if eol_img.tol_id not in eol_ids_lookup:
                print(f"Can't find the tol_id {eol_img.tol_id}")
                logger.warning(
                    "EvoBio10m ID missing. [tol_id: %s]",
                    eol_img.tol_id,
                )
                continue

            # fetching page ID
            content_id, page_id = eol_ids_lookup[eol_img.tol_id]
            global_id = eol_img.tol_id
            
            # checking for global id in split
            if global_id not in splits[args.split] or global_id in finished_ids:
                # logger.info(
                #     "Skipping global ID already in split or finished. [global_id: %s, split: %s]",
                #     global_id,
                #     args.split,
                # )
                continue
            
            # checking for page id
            if page_id not in name_lookup:
                logger.warning(
                    "Page ID missing in name lookup. [page_id: %s]", page_id
                )
                continue
            
            # using name lookup for taxon, common, classes
            taxon, common, classes = name_lookup[page_id]

            if taxon.scientific in species_blacklist:
                logger.info(
                    "Skipping species in blacklist. [scientific: %s]", taxon.scientific
                )
                continue

            # Check if taxon has complete taxonomy
            if not has_complete_taxonomy(taxon):
                logger.info(
                    "Skipping taxon without complete taxonomy. [scientific: %s]",
                    taxon.scientific,
                )
                continue

            # Create folder name
            folder_name = create_taxonomic_folder_name(taxon)
            if not folder_name:
                continue

            file = tar.extractfile(member)
            try:
                img = load_img(file)
            except OSError as err:
                logger.warning(
                    "Error opening file. Skipping. [tar: %s, err: %s]", imgset_path, err
                )
                continue

            # Save image to folder
            save_image_to_folder(img, global_id, folder_name, "EOL", taxon, common)


########
# INAT21
########


def copy_inat21_from_clsdir(clsdir):
    logger = logging.getLogger(f"p{os.getpid()}")

    db = evobio10m_reproduce.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, filename, cls_num FROM inat21;"
    evobio10m_id_lookup = {
        (filename, cls_num): evobio10m_id
        for evobio10m_id, filename, cls_num in db.execute(select_stmt).fetchall()
    }
    db.close()

    name_lookup = naming_reproduce.load_name_lookup(disk_reproduce.inat21_name_lookup_json, keytype=int)

    clsdir_path = os.path.join(disk_reproduce.inat21_root_dir, clsdir)
    for i, filename in enumerate(os.listdir(clsdir_path)):
        filepath = os.path.join(clsdir_path, filename)

        cls_num, *_ = clsdir.split("_")
        cls_num = int(cls_num)

        if (filename, cls_num) not in evobio10m_id_lookup:
            logger.warning(
                "Evobio10m ID missing. [image: %s, cls: %d]", filename, cls_num
            )
            continue

        global_id = evobio10m_id_lookup[(filename, cls_num)]
        if global_id not in splits[args.split] or global_id in finished_ids:
            continue

        taxon, common, classes = name_lookup[cls_num]

        if taxon.scientific in species_blacklist:
            continue

        # Check if taxon has complete taxonomy
        if not has_complete_taxonomy(taxon):
            continue

        # Create folder name
        folder_name = create_taxonomic_folder_name(taxon)
        if not folder_name:
            continue

        img = load_img(filepath)
        # Save image to folder
        save_image_to_folder(img, global_id, folder_name, "iNat21", taxon, common)


#########
# BIOSCAN
#########


def copy_bioscan_from_part(part):
    logger = logging.getLogger(f"p{os.getpid()}")

    db = evobio10m_reproduce.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, part, filename FROM bioscan;"
    evobio10m_id_lookup = {
        (part, filename): evobio10m_id
        for evobio10m_id, part, filename in db.execute(select_stmt).fetchall()
    }
    db.close()

    name_lookup = naming_reproduce.load_name_lookup(disk_reproduce.bioscan_name_lookup_json)

    partdir = os.path.join(disk_reproduce.bioscan_root_dir, f"part{part}")
    for i, filename in enumerate(os.listdir(partdir)):
        if (part, filename) not in evobio10m_id_lookup:
            logger.warning(
                "EvoBio10m ID missing. [part: %d, filename: %s]", part, filename
            )
            continue

        global_id = evobio10m_id_lookup[(part, filename)]
        if global_id not in splits[args.split] or global_id in finished_ids:
            continue

        taxon, common, classes = name_lookup[global_id]

        if taxon.scientific in species_blacklist:
            continue

        # Check if taxon has complete taxonomy
        if not has_complete_taxonomy(taxon):
            continue

        # Create folder name
        folder_name = create_taxonomic_folder_name(taxon)
        if not folder_name:
            continue

        filepath = os.path.join(partdir, filename)
        img = load_img(filepath)
        # Save image to folder
        save_image_to_folder(img, global_id, folder_name, "BIOSCAN", taxon, common)


######
# MAIN
######


def check_status():
    finished_ids = set()
    for root, dirs, files in os.walk(outdir):
        for file in files:
            if file.endswith(".jpg"):
                # Extract global_id from filename (remove .jpg extension)
                global_id = file[:-4]
                finished_ids.add(global_id)
    return finished_ids


sentinel = "STOP"


def worker(input):
    logger = logging.getLogger(f"p{os.getpid()}")
    for func, args in iter(input.get, sentinel):
        logger.info(f"Started {func.__name__}({', '.join(map(str, args))})")
        func(*args)
        logger.info(f"Finished {func.__name__}({', '.join(map(str, args))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width", type=int, default=256, help="Width of resized images."
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Height of resized images."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "train_small"], default="val"
    )
    parser.add_argument("--tag", default="dev", help="The suffix for the directory.")
    #FIXME: Currently, does not work with multiple processes.
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of processes to use."
    )
    args = parser.parse_args()

    # Set up some global variables that depend on CLI args.
    resize_size = (args.width, args.height)
    outdir = f"{evobio10m_reproduce.get_outdir(args.tag)}/{args.width}x{args.height}/{args.split}_folders"
    os.makedirs(outdir, exist_ok=True)
    print(f"Writing images to {outdir}.")

    # Initialize CSV file
    csv_path = os.path.join(outdir, "image_metadata.csv")
    
    # Create CSV header
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['global_id', 'folder_name', 'image_filename', 'source_dataset', 'relative_path', 'scientific_name', 'common_name', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'])

    # db_path = f"{evobio10m_reproduce.get_outdir(args.tag)}/mapping.sqlite"
    db_path = os.path.abspath(disk_reproduce.db)
    print(f"Using database at {db_path}.")

    # Load train/val/train_small splits
    splits = evobio10m_reproduce.load_splits(db_path)

    # Load images already written to avoid duplicate work.
    finished_ids = check_status()
    rootlogger.info("Found %d finished examples.", len(finished_ids))

    # Load image and species blacklists for rare species
    image_blacklist, species_blacklist = load_blacklists()

    # All jobs read from this queue
    task_queue = multiprocessing.Queue()

    # Submit all tasks
    # EOL
    for imgset_name in sorted(os.listdir(disk_reproduce.eol_root_dir)):
        assert imgset_name.endswith(".tar.gz")
        imgset_path = os.path.join(disk_reproduce.eol_root_dir, imgset_name)
        task_queue.put((copy_eol_from_tar, (imgset_path,)))

    # Bioscan
    # 113 parts in bioscan
    for i in range(1, 114):
        task_queue.put((copy_bioscan_from_part, (i,)))

    # iNat
    for clsdir in os.listdir(disk_reproduce.inat21_root_dir):
        task_queue.put((copy_inat21_from_clsdir, (clsdir,)))

    processes = []
    # Start worker processes
    for i in range(args.workers):
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        processes.append(p)
        p.start()

    # Stop worker processes
    for i in range(args.workers):
        task_queue.put(sentinel)

    for p in processes:
        p.join()

    print(f"CSV metadata saved to {csv_path}")
