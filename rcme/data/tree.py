import random

# Sample file paths
file_paths = [
    "00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris",
    "00001_Animalia_Chordata_Mammalia_Primates_Hominidae_Homo_sapiens",
    "00002_Animalia_Arthropoda_Insecta_Lepidoptera_Papilionidae_Papilio_machaon",
    "00003_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris",
    "00004_Animalia_Chordata_Mammalia_Carnivora_Felidae_Panthera_tigris",
    "00004_Animalia_Chordata_Mammalia_Carnivora_Felidae_Pant_tigris",
    "00004_Plantae_Chordata_Mammalia_Carnivora_Felidae_Panthera_tigris",
    "00005_Plantae_Chordata_Mammalia_Carnivora_Felidae_Panthera_tigris"
]

def random_sample_by_rank(file_paths, query_text, rank_index):
    # Step 1: Filter file paths that match the rank and exclude the target value if needed
    if rank_index!=0:
        valid_paths = [path for path in file_paths if path.split("_")[1:][rank_index]==query_text[rank_index] and path.split("_")[1:][rank_index-1]==query_text[rank_index-1]]
    else:
        valid_paths = [path for path in file_paths if path.split("_")[1:][rank_index]==query_text[rank_index]]
    random_path = random.choice(valid_paths)
    if rank_index!=0:
        valid_paths_negative = [path for path in file_paths if path.split("_")[1:][rank_index-1]==query_text[rank_index-1] and path.split("_")[1:][rank_index]!=query_text[rank_index]]
    else:
        valid_paths_negative = [path for path in file_paths if path.split("_")[1:][0]!=query_text[0]]
    if valid_paths_negative:
        random_path_negative = random.choice(valid_paths_negative)
    elif not valid_paths_negative:
        _, random_path_negative = random_sample_by_rank(file_paths, query_text, rank_index-1)
    return random_path, random_path_negative

if __name__ == "__main__":
    # Define the rank index (e.g., 2 = "Phylum", 5 = "Genus") and the target rank value (e.g., "Annelida")
    rank_index = 1  # Phylum index
    target_rank_value = "Animalia Chordata Mammalia Carnivora Felidae Panthera tigris"  # The rank value you are interested in (e.g., "Annelida")

    # Get a random path that matches the rank and has a different species
    sampled_path = random_sample_by_rank(file_paths, target_rank_value.split(" "), rank_index)

    print(f"Sampled Path: {sampled_path}")