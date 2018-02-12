import multiprocessing
from tqdm import tqdm

def parallel(propertyClass, filesname):
    """parallelize calls to an external executable

            Parameters
            ----------
            propertyClass: X-ray property class

            filesname: :pdb_file

            Returns
            -------
            profiles: : list of properties profiles

        """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        profiles = list(tqdm(pool.imap(propertyClass().from_crysol_pdb, filesname),total=len(filesname)))
