"""
Script for generating and subsetting the JUMP1 and LINCS datasets, and generating the necessary pickle files
All necessary pickle files and CSVs are provided in the repo 
(Note that some pickles used for the study were generated with stochasticity and a non-fixed random seed, so will not be exactly the same)

We'll need two repos: 
https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted
https://github.com/broadinstitute/lincs-cell-painting/

# Plate barcode -> plate map name: https://github.com/broadinstitute/lincs-cell-painting/blob/master/metadata/platemaps/2016_04_01_a549_48hr_batch1/barcode_platemap.csv
# plate map name -> broad sample: https://github.com/broadinstitute/lincs-cell-painting/tree/master/metadata/platemaps/2016_04_01_a549_48hr_batch1/platemap
# broad sample -> MOA: 
#     https://github.com/broadinstitute/lincs-cell-painting/blob/f865c796757326dcc377e30dc5b11b5ae392a98b/metadata/moa/repurposing_info_external_moa_map_resolved.tsv
#     https://github.com/broadinstitute/lincs-cell-painting/blob/f865c796757326dcc377e30dc5b11b5ae392a98b/metadata/moa/repurposing_info.tsv
# broad sample -> MOA and target:
#     lincs-cell-painting/metadata/moa/repurposing_info_long.tsv
"""
from classification import *

def convertListToString(l, delimeter="|"):
    s = ''
    for i in range(0, len(l)):
        if i == len(l) - 1:
            s += l[i]
        else:
            s += l[i] + delimeter
    return s

def instantiateJumpDictionaries():
    """
    Create dictionaries from 2021_Chandrasekaran_submitted repo
    """
    base_dir = "/home/wongd26/workspace/2021_Chandrasekaran_submitted/"
    barcode_to_pert_type = {}
    df = pd.read_csv(base_dir + "benchmark/output/experiment-metadata.tsv", sep="\t")
    for index, row in df.iterrows():
        barcode_to_pert_type[row["Assay_Plate_Barcode"]] = row["Perturbation"], row["Cell_type"]
    pickle.dump(barcode_to_pert_type, open("pickles/JUMP1/barcode_to_pert_type.pkl", "wb"))
    batch_pert_well_to_sample = {}
    for batch in os.listdir(base_dir + "metadata/platemaps/"):
        for textfile in os.listdir(base_dir + "metadata/platemaps/" + batch + "/platemap/"):
            if "compound" in textfile: 
                pert = "compound"
            if "crispr" in textfile:
                pert = "crispr"
            if "orf" in textfile:
                pert = "orf"
            df = pd.read_csv(base_dir + "metadata/platemaps/" + batch + "/platemap/" + textfile, sep="\t")
            for index, row in df.iterrows():
                if isinstance(row["broad_sample"], float): ##if no broad_sample - found in crispr and orf platemaps
                    batch_pert_well_to_sample[batch, pert, row["well_position"]] = "no_sample"
                else:
                    batch_pert_well_to_sample[(batch, pert, row["well_position"])] = row["broad_sample"]
    pickle.dump(batch_pert_well_to_sample, open("pickles/JUMP1/batch_well_to_sample.pkl", "wb"))
    sample_to_gene_target = {} 
    for textfile in os.listdir(base_dir + "/metadata/external_metadata/"):
        if "targets" in textfile: ##skip JUMP-Target-1_compound_metadata_targets.tsv
            continue 
        df = pd.read_csv(base_dir + "/metadata/external_metadata/" + textfile, sep="\t")
        for index, row in df.iterrows():
            if isinstance(row["control_type"], float): ##if control type is empty instance (most cases) this is an experimental condition targeting a gene
                sample_to_gene_target[row["broad_sample"]] = row["gene"], "experimental"
            elif isinstance(row["broad_sample"], float): ##if no Broad sample, then no target (just happens in compound metadata in tsvs, but there are crispr and orf wells that don't have a sample...) -- is this valid? For compounds this indicates a negcon type. But for crispr - what to make of the control_type? rn we are calling it a negcon...
                sample_to_gene_target["no_sample"] = "no_target", row["control_type"]
            elif isinstance(row["gene"], float): ##if no gene target (but we have a broad sample, i.e. see CRISPR metadata sheet)
                sample_to_gene_target[row["broad_sample"]] = "no_target", row["control_type"]
            else: 
                sample_to_gene_target[row["broad_sample"]] = row["gene"], row["control_type"]
    pickle.dump(sample_to_gene_target, open("pickles/JUMP1/sample_to_gene_target.pkl", "wb"))
    
def parseJumpImageNames():
    """
    Writes a dictionary of key: imagename (full path) to value: tuple of (pert_type, broad sample, and gene target)
    """
    ##given full path, extract pert type
    ## extract well position, and use maps to get sample and genetic target
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    row_map = {i: letters[i - 1] for i in range(1, len(letters)+1)}
    batch_pert_well_to_sample = pickle.load( open("pickles/JUMP1/batch_well_to_sample.pkl", "rb"))
    barcode_to_pert_type = pickle.load( open("pickles/JUMP1/barcode_to_pert_type.pkl", "rb"))
    sample_to_gene_target = pickle.load(open("pickles/JUMP1/sample_to_gene_target.pkl", "rb"))
    image_map = {}
    image_dir = "data/CPJUMP1/"
    for batch in os.listdir(image_dir):
        for sub_batch in os.listdir(image_dir + batch + "/images/"):
            barcode = sub_batch[sub_batch.find("BR"):sub_batch.find("BR") + 10]
            if not os.path.isdir(image_dir + batch + "/images/" + sub_batch):
                continue
            for image in os.listdir(image_dir + batch + "/images/" + sub_batch + "/Images/"):
                if ".tiff" in image:
                    row = row_map[int(image[1:3])]
                    column = image[4:6]
                    pert_type, cell_type = barcode_to_pert_type[barcode]
                    broad_sample = batch_pert_well_to_sample[(batch, pert_type, row + column)]
                    gene_target, control_type = sample_to_gene_target[broad_sample]
                    image_map[image_dir + batch + "/images/" + sub_batch + "/Images/" + image] = (broad_sample, pert_type, cell_type, gene_target, control_type)
    pickle.dump(image_map, open("pickles/JUMP1/image_map.pkl", "wb"))

def createJumpImageCSV():
    """
    Creates CSVs with headers: imagename (full path), broad_sample, pert_type, gene_target
    full CSV of all pertubations: all_images.csv
    and subsets: {perturbation}_images.csv, {perturbation}_images_control.csv
    """
    image_map = pickle.load(open("pickles/JUMP1/image_map.pkl", "rb"))
    df = pd.DataFrame()
    imagenames, samples, perturbations, cell_types, gene_targets, control_types = [], [], [], [], [], []
    for key in image_map:
        if "ch1" not in key: ##kust keep ch1 for CSVs so that we have unique field of views in the CSVs
            continue
        imagenames.append(key)
        samples.append(image_map[key][0])
        perturbations.append(image_map[key][1])
        cell_types.append(image_map[key][2])
        gene_targets.append(image_map[key][3])
        control_types.append(image_map[key][4])
    df["imagename"] = imagenames
    df["broad_sample"] = samples
    df["perturbation"] = perturbations
    df["cell_type"] = cell_types
    df["gene_targets"] = gene_targets
    df["control_type"] = control_types
    df.to_csv("csvs/JUMP1/all_images.csv", index=False)
    ##create CSVs for each perturbation type 
    for pert in ["compound", "crispr", "orf"]:
        df_sub = df[df["perturbation"] == pert]
        df_sub.to_csv("csvs/JUMP1/{}_images.csv".format(pert), index=False)
        df_sub_sub = df_sub[df_sub["control_type"].isin(["negcon", "poscon_cp"])] ##if we want just neg and pos controls
        df_sub_sub.to_csv("csvs/JUMP1/{}_images_control.csv".format(pert), index=False)
        df_sub_sub_sub = df_sub_sub[df_sub_sub["control_type"] != "negcon"]
        df_sub_sub_sub.to_csv("csvs/JUMP1/{}_images_control_no_neg.csv".format(pert), index=False)
    ##create ones for just compound and crispr
    df_compound_and_crispr = df[df["perturbation"].isin(["compound", "crispr"])]
    df_compound_and_crispr.to_csv("csvs/JUMP1/compound_and_crispr_images.csv", index=False)
    df_compound_and_crispr_control = df_compound_and_crispr[df_compound_and_crispr["control_type"].isin(["negcon", "poscon_cp"])]
    df_compound_and_crispr_control.to_csv("csvs/JUMP1/compound_and_crispr_images_control.csv", index=False)

def createJUMP1CompoundMOACSV():
    """
    Creates JUMP1 csv with MOA annotations: csvs/JUMP1/compound_images_with_MOA.csv
    also no polypharm subset compound_images_with_MOA_no_polypharm.csv 
    """
    broad_sample_to_compound = pickle.load(open("pickles/JUMP1/broad_sample_to_compound_map.pkl", "rb"))
    compound_to_moa = pickle.load(open("pickles/JUMP1/compound_to_moa_map.pkl", "rb"))
    df = pd.read_csv("csvs/JUMP1/compound_images.csv")
    moa_list = []
    compound_list = []
    for index, row in df.iterrows():
        sample = row["broad_sample"]
        compound = broad_sample_to_compound[sample]
        compound_list.append(compound)
        moas = compound_to_moa[compound]
        moa_list.append(convertListToString(list(moas)))
    df["perturbation"] = compound_list
    df["moas"] = moa_list
    unknown_compounds = ['trometamol', 'carzenide', '1-octanol', 'glutamine-(l)', 'L-Cystine', 'pidolic-acid', "cromakalim"]
    df = df[~df["perturbation"].isin(unknown_compounds)]
    df = df[["imagename","broad_sample", "perturbation", "cell_type", "gene_targets", "control_type", "perturbation", "moas"]]
    df.to_csv("csvs/JUMP1/compound_images_with_MOA.csv", index=False)
    ##no polypharm df 
    df_no_poly = df[~df["moas"].str.contains("|", regex=False)]
    df_no_poly.to_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm.csv", index=False)
    df_no_poly_no_neg = df_no_poly[df_no_poly["moas"] != "Empty"]
    df_no_poly_no_neg.to_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm_no_neg.csv", index=False)

def createJUMP1CompoundMOACSVWellExcluded():
    """
    exclude plate BR00116995 because it was QC'ed out 
    """
    df = pd.read_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm.csv")
    df = df[~df["imagename"].str.contains("BR00116995", regex=False)]
    df.to_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv", index=False)

def createCompoundHoldout(study="JUMP1", min_compounds=2):
    if study == "JUMP1":
        df = pd.read_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv")
    if study == "lincs":
        df = pd.read_csv("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv")
    poly_moas_map = pickle.load(open("pickles/{}/poly_compound_moas_map.pkl".format(study), "rb"))
    holdout_compounds = []
    for moa in poly_moas_map:
        holdout_compounds.append(poly_moas_map[moa].pop())
    test_df = df[df["perturbation"].isin(holdout_compounds)] 
    test_df = test_df.copy()
    test_df_no_neg = test_df[test_df["moas"].ne("Empty")]
    test_df_no_neg = test_df_no_neg.copy()
    if study == "JUMP1":
        test_df.to_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_{}.csv".format(min_compounds), index=False)
        test_df_no_neg.to_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_test_{}_no_neg.csv".format(min_compounds), index=False)
    if study == "lincs":
        test_df.to_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_test_{}.csv".format(min_compounds), index=False)
        test_df_no_neg.to_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_test_{}_no_neg.csv".format(min_compounds), index=False)
    other_df = df[~df["perturbation"].isin(holdout_compounds)] 
    other_df = other_df.copy()
    createMOADistributedTrainValCSVs(study=study, df=other_df, percent_split=.70, min_compounds=min_compounds)

def createMOADistributedTrainValCSVs(study=None, df=None, percent_split=None, min_compounds=None):
    """
    Will take a dataframe, and write train and validation CSVs such that each gene is represented in both, and wells are split by percent_split
    Saves to csvs/learning/{split}_split/ directory
    """
    ##create map from gene to plate_wells
    plate_wells = []
    label_to_well = {label: set() for label in set(df["moas"])} ##moa to set of plate_wells, value will later become dictionary 
    for index, row in df.iterrows():
        plate_well = getBarcode(row["imagename"]) + getRowColumn(row["imagename"])
        label_to_well[row["moas"]].add(plate_well)
        plate_wells.append(plate_well)
    df["plate_well"] = plate_wells
    label_to_well = {key: list(label_to_well[key]) for key in label_to_well} ##convert set to list
    ##change label_to_well such that values are now dictionaries with keys: train, validation, instead of just lists
    for label in label_to_well:
        wells = label_to_well[label]
        ##randomize list of plate_wells
        random.shuffle(wells)
        train_wells = wells[0:int(percent_split * len(wells))]
        validation_wells = wells[int(percent_split * len(wells)):]
        if len(validation_wells) == 0:
            print("ZZZ moa has no validation wells: ", label)
        label_to_well[label] = {"train": train_wells, "validation": validation_wells} 
    ##now iterate over df row by row and make the train, val dataframes using info from label_to_well
    df_train = pd.DataFrame(columns=df.columns)
    df_validation = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        df_new_row = pd.DataFrame({key: row[key] for key in df.columns}, index=[0])
        identifier = row["plate_well"]
        if identifier in label_to_well[row["moas"]]["train"]:
            df_train = pd.concat([df_train, df_new_row], ignore_index=True)
        elif identifier in label_to_well[row["moas"]]["validation"]:
            df_validation = pd.concat([df_validation, df_new_row], ignore_index=True)
        else:
            assert("identifier not found in any of train/val/test partition")
    del df_train["plate_well"]
    del df_validation["plate_well"]
    ##make sure each moa is represented in train, val
    assert(len(set(df_train["moas"])) == len(set(df_validation["moas"])))
    if study == "JUMP1":
        df_train.to_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_train_{}.csv".format(min_compounds), index=False)
        df_validation.to_csv("csvs/JUMP1/learning/moas/compound_images_with_MOA_no_polypharm_well_excluded_compound_holdout_valid_{}.csv".format(min_compounds), index=False)
    if study == "lincs":
        df_train.to_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_train_{}.csv".format(min_compounds), index=False)
        df_validation.to_csv("csvs/lincs/learning/moas/lincs_ten_micromolar_no_polypharm_compound_holdout_valid_{}.csv".format(min_compounds), index=False)

def instantiateLINCSDictionaries():
    """
    Create dictionaries from data/lincs-cell-painting/
    """
    base_dir = "data/lincs-cell-painting/"
    ##create barcode_to_platemap
    barcode_to_platemap = {}
    df = pd.read_csv(base_dir + "metadata/platemaps/2016_04_01_a549_48hr_batch1/barcode_platemap.csv")
    for index, row in df.iterrows():
        barcode_to_platemap[row["Assay_Plate_Barcode"]] = row["Plate_Map_Name"]
    pickle.dump(barcode_to_platemap, open("pickles/lincs/lincs_barcode_to_platemap.pkl", "wb"))
    ##create platemap_well_to_sample_and_concentration
    platemap_well_to_sample_and_concentration = {}
    for platemap in os.listdir(base_dir + "metadata/platemaps/2016_04_01_a549_48hr_batch1/platemap/"):
        df = pd.read_csv(base_dir + "metadata/platemaps/2016_04_01_a549_48hr_batch1/platemap/" + platemap, sep="\t")
        for index, row in df.iterrows():
            r = row["well_position"][0]
            col = row["well_position"][1:]
            if isinstance(row["broad_sample"], float):
                platemap_well_to_sample_and_concentration[row["plate_map_name"] + "_" + r + col] = "NoSample", "Empty"
            else:
                platemap_well_to_sample_and_concentration[row["plate_map_name"] + "_" + r + col] = row["broad_sample"], row["mmoles_per_liter"]
    pickle.dump(platemap_well_to_sample_and_concentration, open("pickles/lincs/lincs_platemap_well_to_sample_and_concentration.pkl", "wb"))
    ##create sample_to_moa_target_pert 
    sample_to_moa_target_pert = {}
    df = pd.read_csv(base_dir + "metadata/moa/repurposing_info_long.tsv", sep="\t")
    for index, row in df.iterrows():
        if row["broad_id"] not in sample_to_moa_target_pert: ##if not in dict, instantiate
            sample_to_moa_target_pert[row["broad_id"]] = {"moa":[], "target":[], "pert_iname":[]}
            sample_to_moa_target_pert[row["deprecated_broad_id"]] = {"moa":[], "target":[], "pert_iname":[]}
            for label in ["moa", "target", "pert_iname"]: ##
                if isinstance(row[label], float):
                    sample_to_moa_target_pert[row["broad_id"]][label] = ["Empty"]
                    sample_to_moa_target_pert[row["deprecated_broad_id"]][label] = ["Empty"]
                else: ##set add to dict entry            
                    sample_to_moa_target_pert[row["broad_id"]][label] = list(set([lab for lab in row[label].split("|")])) 
                    sample_to_moa_target_pert[row["deprecated_broad_id"]][label] = list(set([lab for lab in row[label].split("|")]))
        else: ##else confirm that new row is matching what is in dictionary 
            for label in ["moa", "target", "pert_iname"]: ##
                if not isinstance(row[label], float):    
                    assert(set([lab for lab in row[label].split("|")]) == set(sample_to_moa_target_pert[row["broad_id"]][label]))
    #iterate over repurposing_info_external_moa_map_resolved.tsv for the samples that are not included in repurposing_info_long.tsv, if there is a collision prioritize repurposing_info_external_moa_map_resolved.tsv (Bornholdt also used this tsv for moa maps)
    df2 = pd.read_csv(base_dir + "metadata/moa/repurposing_info_external_moa_map_resolved.tsv", sep="\t")
    for index, row in df2.iterrows():
        if row["broad_sample"] not in sample_to_moa_target_pert: ##if not present instantiate
            sample_to_moa_target_pert[row["broad_sample"]] = {"moa":[], "target":[], "pert_iname":[]}
            for label in ["moa", "pert_iname"]:
                if isinstance(row[label], float): ##if N/A then set as N/A
                    sample_to_moa_target_pert[row["broad_sample"]][label] = ["Empty"]
                else:
                    sample_to_moa_target_pert[row["broad_sample"]][label] = [lab for lab in row[label].split("|")]
            sample_to_moa_target_pert[row["broad_sample"]]["target"] = ["Empty"] #if wasn't found in repurposing_info_long.tsv and because no targets available in this TSV
        else: #already exists in repurposing_info_long.tsv
            for label in ["moa", "target", "pert_iname"]:
                old = sample_to_moa_target_pert[row["broad_sample"]][label]
                if isinstance(row[label], float): ##if new moa is N/A stick with old
                    continue
                new = set([lab for lab in row[label].split("|")])
                ##replace old with new most recent
                sample_to_moa_target_pert[row["broad_sample"]][label] = list(new)
    pickle.dump(sample_to_moa_target_pert, open("pickles/lincs/lincs_sample_to_moa_target_pert.pkl", "wb"))
        
def parseLINCSImageName():
    """
    Writes a dictionary of key: imagename (full path) to value: {"moa":[], "target":[]}
    """
    ##given full path, extract moas and targets
    ## extract well position, and use maps to get info
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    row_map = {i: letters[i - 1] for i in range(1, len(letters)+1)}
    barcode_to_platemap = pickle.load(open("pickles/lincs/lincs_barcode_to_platemap.pkl", "rb"))
    platemap_well_to_sample_and_concentration = pickle.load(open("pickles/lincs/lincs_platemap_well_to_sample_and_concentration.pkl", "rb"))
    sample_to_moa_target_pert = pickle.load(open("pickles/lincs/lincs_sample_to_moa_target_pert.pkl", "rb"))
    image_map = {}
    image_dir = "data/LINCSDatasetCompressed/"
    for batch in os.listdir(image_dir):
        for image in os.listdir(image_dir + batch):
            if ".png" in image:
                #these two pngs are incomplete for the compressed LINCS dataset...
                if (batch == "SQ00015168" and "r10c07f06p01" in image) or (batch == "SQ00015168" and "r10c07f05p01" in image): 
                    continue
                barcode = batch
                row = row_map[int(image[1:3])]
                column = image[4:6]
                platemap = barcode_to_platemap[barcode]
                sample, concentration = platemap_well_to_sample_and_concentration[platemap + "_" + row + column]
                if sample == "NoSample": ##DMSO Control 
                    image_map[image_dir + batch + "/" + image] = {"moa":["Empty"], "target":["Empty"],  "pert_iname":["Empty"], "concentration": "Empty"}
                else:
                    moas, targets, pert_iname = sample_to_moa_target_pert[sample]["moa"], sample_to_moa_target_pert[sample]["target"], sample_to_moa_target_pert[sample]["pert_iname"]
                    assert(len(pert_iname) == 1)
                    image_map[image_dir + batch + "/" + image] = {"moa":moas, "target":targets, "concentration":concentration, "pert_iname": pert_iname}
    pickle.dump(image_map, open("pickles/lincs/lincs_image_map.pkl", "wb"))

def createLINCSImageCSV():
    image_map = pickle.load(open("pickles/lincs/lincs_image_map.pkl", "rb"))
    df = pd.DataFrame()
    imagenames, gene_targets, moas, concentrations, perturbations = [], [], [], [], []
    for key in image_map:
        if "ch1" not in key: ##kust keep ch1 for CSVs so that we have unique field of views in the CSVs
            continue
        imagenames.append(key)
        targets_list = image_map[key]["target"]
        targets_as_string = convertListToString(targets_list, delimeter="|")
        moas_list = image_map[key]["moa"]
        moas_as_string = convertListToString(moas_list, delimeter="|")
        gene_targets.append(targets_as_string)
        moas.append(moas_as_string)
        concentrations.append(image_map[key]["concentration"])
        perturbations.append(image_map[key]["pert_iname"][0])
    df["imagename"] = imagenames
    df["gene_targets"] = gene_targets
    df["moas"] = moas
    df["concentration"] = concentrations
    df["perturbation"] = perturbations
    df.to_csv("csvs/lincs/lincs_all_images.csv", index=False)
    ##create sub dataframes at 10uM and no polypharmacology
    df_control = df[df["concentration"] == "Empty"]
    df_perturbation =  df[df["concentration"] != "Empty"]
    df_perturbation_ten = df_perturbation[df_perturbation["concentration"].between(9.9, 10.1, inclusive=False)]
    df_ten = pd.concat([df_control, df_perturbation_ten],ignore_index=True)
    df_ten.to_csv("csvs/lincs/lincs_ten_micromolar.csv", index=False)
    df_ten_no_poly = df_ten[~df_ten["moas"].str.contains("|", regex=False)]
    df_ten_no_poly.to_csv("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv", index=False)
    ##create singleton pharmacology dataset
    df_singleton = df[~df["moas"].str.contains("|", regex=False)]
    df_singleton.to_csv("csvs/lincs/lincs_no_polypharm.csv", index=False)

def createJUMPCellAndTimeMap():
    base_dir = "/home/wongd26/workspace/2021_Chandrasekaran_submitted/"
    df = pd.read_csv(base_dir + "benchmark/output/experiment-metadata.tsv", sep="\t")
    mapp = {} ##key: batch|plate, value: (cell type, time point)
    for index, row in df.iterrows(): ##Batch	Plate_Map_Name	Assay_Plate_Barcode	Perturbation	Cell_type	Time	Density	Antibiotics	Cell_line	Time_delay	Times_imaged
        if row["Batch"] + "|" + row["Assay_Plate_Barcode"] not in mapp:
            mapp[row["Batch"] + "|" + row["Assay_Plate_Barcode"]] = (row["Cell_type"], row["Time"])
        else:
            assert(mapp[row["Batch"] + "|" + row["Assay_Plate_Barcode"]] == (row["Cell_type"], row["Time"]))
    pickle.dump(mapp, open("pickles/JUMP1/barcode_to_cell_type_and_time.pkl", "wb"))

def getMOAMap():
    base_dir = "/home/wongd26/workspace/2021_Chandrasekaran_submitted/"
    df = pd.read_csv(base_dir + "benchmark/input/JUMP-Target-1_compound_metadata_additional_annotations.tsv", sep="\t")
    broad_sample_to_compound = {sample: "" for sample in set(df["broad_sample"])}
    compound_to_moa = {compound: set() for compound in set(df["pert_iname"])}
    moa_set = set()
    for index, row in df.iterrows():
        broad_sample_to_compound[row["broad_sample"]] = row["pert_iname"]
        if isinstance(row["moa_list"], float):
            compound_to_moa[row["pert_iname"]].add("Empty") 
            continue
        compound_to_moa[row["pert_iname"]] = compound_to_moa[row["pert_iname"]].union(set(row["moa_list"].split("|")))
        moa_set = moa_set.union(set(row["moa_list"].split("|")))
    ## get rid of empties, missing moa for VEGFA
    compound_to_moa = {key: compound_to_moa[key] for key in compound_to_moa if compound_to_moa[key] != set()} 
    broad_sample_to_compound["no_sample"] = "DMSO"
    compound_to_moa["DMSO"] = set(["Empty"]) ##replace DMSO's 'control vehicle' MOA with Empty 
    pickle.dump(broad_sample_to_compound, open("pickles/JUMP1/broad_sample_to_compound_map.pkl", "wb"))
    pickle.dump(compound_to_moa, open("pickles/JUMP1/compound_to_moa_map.pkl", "wb"))

def getLINCSMOAMap():
    df = pd.read_csv("csvs/lincs/lincs_no_polypharm.csv")
    compound_to_moa_map = {}
    for index, row in df.iterrows():
        if row["perturbation"] not in compound_to_moa_map:
            compound_to_moa_map[row["perturbation"]] = {row["moas"]} ##cast as set to stay consistent with pickles/JUMP/compound_to_moa_map.pkl
        else:
            assert compound_to_moa_map[row["perturbation"]] == {row["moas"]}
    pickle.dump(compound_to_moa_map, open("pickles/lincs/compound_to_moa_map.pkl", "wb"))

def getChannelStats(study="JUMP1"):
    """
    Calculates the global mean and stds for each image channel (scaled between 0 and 1), 
    technically std will be a close proxy (mean of individual image stds)
    Writes pickle file containing stats to pickles/{STUDY}/
    """
    if study == "JUMP1":
        label_index_map = pickle.load(open("pickles/JUMP1/label_index_map_from_compound_images_with_MOA_no_polypharm.csv.pkl", "rb"))
        images = JUMPMOADataset("csvs/JUMP1/compound_images_with_MOA_no_polypharm.csv", None, label_index_map=label_index_map)
    if study == "lincs":
        label_index_map = pickle.load(open("pickles/lincs/label_index_map_from_lincs_ten_micromolar_no_polypharm.csv.pkl", "rb"))
        images = LINCSClassificationDataset("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv", None, label_index_map=label_index_map)
    sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(0, len(images))))
    generator = torch.utils.data.DataLoader(images, sampler=sampler, batch_size=36, num_workers=32)
    stats = { "mean": {i: [] for i in range(0, 5)}, "std": {i: [] for i in range(0, 5)} }
    image_counter = 0
    for imagenames, image_stack, _, _ in generator:
        image_stack = image_stack.detach().cpu().numpy()
        for i in range(0, image_stack.shape[0]): ##iterate over batch
            stack_b = image_stack[i]
            valid = True
            for j in range(0, stack_b.shape[0]): ##iterate over image channels 
                channel_j = stack_b[j,:]
                max_j = np.amax(channel_j)
                min_j = np.amin(channel_j)
                channel_j = (channel_j - min_j) / (max_j + .000001)
                mean = np.mean(channel_j)
                std = np.std(channel_j)
                if not math.isnan(mean) and not math.isnan(std):
                    stats["mean"][j].append(mean)
                    stats["std"][j].append(std)
                else:
                    print("invalid value encountered!")
                    valid = False 
            if valid:
                image_counter += 1
    for key in stats:
        stats[key] = {key2: np.mean(stats[key][key2]) for key2 in stats[key].keys()}
    if study == "JUMP1":
        pickle.dump(stats, open("pickles/{}/channel_stats_compounds_raw.pkl".format(study), "wb"))
    if study == "lincs":
        pickle.dump(stats, open("pickles/{}/compressed_channel_stats_raw_corrected.pkl".format(study), "wb"))

def getFullBarcodePlate():
    """
    Instantiate dictionary with key: batch|abbreviated barcode to value: full plate name,
    saves to pickle "pickles/JUMP1/fullBarcodePlateNameMap.pkl" 
    """
    batch_barcode_dictionary = {}
    for batch in os.listdir("data/CPJUMP1/"):
        for plate in os.listdir("data/CPJUMP1/" + batch + "/images/"):
            if "BR" not in plate:
                continue
            abbrev_plate = getBarcode(plate)
            if "{}|{}".format(batch, abbrev_plate) not in batch_barcode_dictionary:
                batch_barcode_dictionary["{}|{}".format(batch, abbrev_plate)] = plate
    pickle.dump(batch_barcode_dictionary, open("pickles/JUMP1/fullBarcodePlateNameMap.pkl", "wb"))

def getLabelIndexMap(csv, study="None", target_column=None):
    """
    Reads CSV with labels found in TARGET_COLUMN, 
    Saves a dictionary from key: label, to value: unique index to pickles/{STUDY}/ directory
    """
    df = pd.read_csv(csv)
    targets = list(set(df[target_column]))
    mapp = {}
    for target in targets:
        mapp[target] = targets.index(target)
    if target_column == "moas" or target_column == "gene_targets":
        pickle.dump(mapp, open("pickles/{}/label_index_map_from_{}.pkl".format(study, csv.replace("csvs/{}/".format(study), "")), "wb"))
    else:
        pickle.dump(mapp, open("pickles/{}/{}_index_map_from_{}.pkl".format(study, target_column, csv.replace("csvs/{}/".format(study), "")), "wb"))

def createRowLetterConverters():
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    row_map = {} ##key letter: value: row as ##, 1-indexed
    for i in range(0, len(letters)):
        if i < 9:
            row_map[letters[i]] = "0" + str(i + 1)
        else:
            row_map[letters[i]] = str(i + 1)
    pickle.dump(row_map, open("pickles/JUMP1/row_letter_to_index.pkl", "wb"))
    row_map = {i: letters[i - 1] for i in range(1, len(letters)+1)}
    pickle.dump(row_map, open("pickles/JUMP1/row_index_to_letter.pkl", "wb"))

def getPolyCompoundMOAs(study="JUMP1"):
    if study == "JUMP1":
        df = pd.read_csv("csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv")
    if study == "lincs":
        df = pd.read_csv("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv")
    moa_to_compound = {}
    for index, row in df.iterrows():
        if row["moas"] not in moa_to_compound:
            moa_to_compound[row["moas"]] = set([row["perturbation"]])
        else:
            moa_to_compound[row["moas"]].add(row["perturbation"])
    moa_to_compound = {moa: moa_to_compound[moa] for moa in moa_to_compound if len(moa_to_compound[moa]) >= 2}
    pickle.dump(moa_to_compound, open("pickles/{}/poly_compound_moas_map.pkl".format(study), "wb"))

def parseJUMPCellProfilerData():
    """
    data/profiles downloaded from JUMP1 repo:
    batch = <BATCH NAME>
    aws s3 cp \
    --no-sign-request \
    --recursive \
    s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/backend/${batch}/ . 

    Parse through well aggregated profiles and save as pickles/JUMP1/cellProfilerFeatures.pkl
    Parse through single cell profiles and save as pickles/JUMP1/cellProfilerSingleCellFeatures.pkl
    """
    profile_map = {} ##key batch + plate barcode + well, value: feature vector
    row_letter_to_index = pickle.load(open("pickles/JUMP1/row_letter_to_index.pkl", "rb"))
    profile_directory = "data/profiles/"
    ##iterate over profiles directory and read CSVs of well-aggregated embeddings 
    for batch in os.listdir(profile_directory):
        if getJumpBatch(batch) == "2020_11_04_CPJUMP1":
            print("started", batch)
            for barcode in os.listdir(profile_directory + batch):
                for f in os.listdir(profile_directory + batch + "/" + barcode):
                    if ".csv" in f:
                        df = pd.read_csv(profile_directory + batch + "/" + barcode + "/" + f)
                        print("started", f)
                        for index,row in df.iterrows():
                            entries = df.loc[index, :].values.tolist()
                            letter_well = entries[1]
                            well_row = row_letter_to_index[letter_well[0]]
                            well_col = letter_well[1:]
                            numeric_well = "r" + well_row + "c" + well_col
                            vector = np.array(entries[2:])
                            key = batch + "|" + entries[0] + "|r" + str(well_row) + "c" + str(well_col) ##batch + plate barcode + well
                            if key not in profile_map:
                                profile_map[key] = vector 
                            else:
                                assert(np.array_equal(profile_map[key], vector))
    pickle.dump(profile_map, open("pickles/JUMP1/cellProfilerFeatures.pkl", "wb"))

def parseSpherizedJUMPCellProfilerData():
    """
    Parses spherized profiles for JUMP, found at: https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted/tree/main/profiles
    """
    directory = "/home/wongd26/workspace/2021_Chandrasekaran_submitted/profiles/2020_11_04_CPJUMP1/"
    for barcode in os.listdir(directory):
        for gz_file in os.listdir(directory + barcode + "/"):
            if ".gz" in gz_file and "spherized" in gz_file:
                with gzip.open(directory + barcode + "/" + gz_file, 'rb') as f_in:
                    with open(directory + barcode + "/" + gz_file.replace(".gz", ""), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    ##iterate over profiles directory and read CSVs of well-aggregated embeddings
    row_letter_to_index = pickle.load(open("pickles/JUMP1/row_letter_to_index.pkl", "rb"))
    profile_map = {} ##key batch + plate barcode + well, value: feature vector
    ##first get minimal set of feature names
    df1 = pd.read_csv(directory + "BR00117017/BR00117017_spherized.csv")
    df1_cols = set(df1.columns)
    index_of_first_feature = list(df1.columns).index("Cells_AreaShape_Area")
    feature_set = set(list(df1.columns)[index_of_first_feature:])
    for barcode in os.listdir(directory):
        batch = getJumpBatch(directory)
        for spherized_csv in os.listdir(directory + barcode + "/"):
            if "spherized" not in spherized_csv or ".csv" not in spherized_csv or ".gz" in spherized_csv:
                continue
            df = pd.read_csv(directory + barcode + "/" + spherized_csv)
            index_of_first_feature = list(df1.columns).index("Cells_AreaShape_Area")
            feature_set = feature_set.intersection(set(list(df.columns)[index_of_first_feature:]))
    feature_set = sorted(list(feature_set)) ##convert to list for fixed ordering
    ##extract features
    for barcode in os.listdir(directory):
        batch = getJumpBatch(directory)
        for spherized_csv in os.listdir(directory + barcode + "/"):
            if "spherized" not in spherized_csv or ".csv" not in spherized_csv or ".gz" in spherized_csv:
                continue
            df = pd.read_csv(directory + barcode + "/" + spherized_csv)
            for index,row in df.iterrows():
                entries = df.loc[index, :].values.tolist()
                letter_well = row["Metadata_Well"]
                well_row = row_letter_to_index[letter_well[0]]
                well_col = letter_well[1:]
                numeric_well = "r" + well_row + "c" + well_col
                vector = [row[feature] for feature in feature_set]
                barcode_in_csv = row["Metadata_Plate"]
                if barcode_in_csv != barcode: ##oddly, some of the CSVs have multiple plates in them...probably a clerical error from the JUMP people 
                    continue
                key = batch + "|" + barcode_in_csv + "|r" + str(well_row) + "c" + str(well_col) ##batch + plate barcode + well
                if key not in profile_map:
                    profile_map[key] = vector 
                else:
                    assert(np.array_equal(profile_map[key], vector))
    pickle.dump(profile_map, open("pickles/JUMP1/cellProfilerFeatures_from_spherized.pkl", "wb"))

def parseDeepProfilerFromModelOutput(study="JUMP1", method="median"):  
    """
    Aggregate DeepProfiler embeddings found in /home/wongd26/workspace/profiler/{study}DeepProfiler/outputs/results/features/ (from running the DeepProfiler model)
    METHOD in ["mean", "median", "pca"]
    Saves pickle file pickles/{study}/deepProfilerFeatures_from_model_{method}.pkl
    https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted/issues/47
    """
    if study == "JUMP1":
        profile_directory = "/home/wongd26/workspace/profiler/JUMP1DeepProfiler/outputs/results/features/"
    if study == "lincs":
        profile_directory = "/home/wongd26/workspace/profiler/LINCSDeepProfiler/outputs/results/features/"
    profile_map = {} ##key batch + plate barcode + well, value: feature vector
    row_letter_to_index = pickle.load(open("pickles/JUMP1/row_letter_to_index.pkl", "rb"))
    barcode_to_platemap = pickle.load(open("pickles/lincs/lincs_barcode_to_platemap.pkl", "rb")) 
    for directory in os.listdir(profile_directory):
        barcode = getBarcode(directory)
        if study == "JUMP1":
            batch = getJumpBatch(directory)
        if study == "lincs":
            batch = barcode_to_platemap[barcode]
        print("started: ", directory)
        nans = 0
        for well in os.listdir(profile_directory + directory):
            well_row = row_letter_to_index[well[0]]
            well_col = well[1:]
            numeric_well = "r" + well_row + "c" + well_col
            single_cells = [] ##single cells for this well 
            for npz in os.listdir(profile_directory + directory + "/" + well): ##iterate over all npz files for this well 
                try:
                    with np.load(profile_directory + directory + "/" + well + "/" + npz, allow_pickle=True) as data: ##keys: metadata, features, locations - each row is a single cell embedding
                        array = data['features'] ##array of many single cell features
                        locations = data['locations'] ##corresponding locations, many single cells
                        for i in range(0, len(array)):
                            if np.any(np.isnan(array[i])): ##if any nans present, skip this single cell embedding 
                                nans += 1
                                continue
                            single_cells.append(array[i])
                except:
                    print("error with: ", profile_directory + directory + "/" + well + "/" + npz)
            if len(single_cells) == 0:
                print("no valid single cells for {} {}, continuing".format(directory, well))
                continue
            single_cells = np.array(single_cells)
            if method == "mean":
                well_embedding = np.mean(single_cells, axis=0)
            if method == "median":
                well_embedding = np.median(single_cells, axis=0)
            if method == "pca":
                well_embedding = PCAAggregate(single_cells)
            profile_map[batch + "|" + barcode + "|" + numeric_well] = well_embedding
    pickle.dump(profile_map, open("pickles/{}/deepProfilerFeatures_from_model_{}.pkl".format(study, method), "wb"))

def parseLINCSCellProfilerData(drop_na="by column", repo="repo_level_3"):
    """
    Parse through well aggregated profiles and save as pickles/lincs/cellProfilerFeatures.pkl, with key: platemap + Metadata_Assay_Plate_Barcode + well, value: feature vector
    If repo == "repo_level_3" will parse https://github.com/broadinstitute/neural-profiling/tree/main/baseline/01_data/ (which was extracted from https://github.com/broadinstitute/lincs-cell-painting/tree/master/profiles)
    If repo == "repo_spherized" will parse https://github.com/broadinstitute/lincs-cell-painting/tree/master/spherized_profiles/profiles
    """
    profile_map = {} ##platemap + Metadata_Assay_Plate_Barcode + well, value: feature vector
    row_letter_to_index = pickle.load(open("pickles/JUMP1/row_letter_to_index.pkl", "rb"))
    dfs = [] ##list to hold pandas dataframes
    ##parse CellProfiler embedding CSVs
    if repo == "repo_level_3":
        df = pd.read_csv("/gpfs/home/wongd26/workspace/profiler/neural-profiling/baseline/01_data/full_level3.csv")
        dfs.append(df)
    if repo == "repo_spherized":
        ##convert .gz to csv
        directory = "/gpfs/workspace/users/wongd26/profiler/lincs-cell-painting/spherized_profiles/profiles/"
        # for gz_file in os.listdir(directory):
        #     if ".gz" in gz_file:
        #         with gzip.open(directory + gz_file, 'rb') as f_in:
        #             with open(directory + gz_file.replace(".gz", ""), 'wb') as f_out:
        #                 shutil.copyfileobj(f_in, f_out)
        df = pd.read_csv(directory + "2016_04_01_a549_48hr_batch1_dmso_spherized_profiles_with_input_normalized_by_whole_plate.csv")
        # df = pd.read_csv(directory + "2016_04_01_a549_48hr_batch1_dmso_spherized_profiles_with_input_normalized_by_dmso.csv") ##appears to be the standard one, but missing a couple wells
        dfs.append(df)
    for df in dfs:
        if drop_na == "by column":
            df = df.dropna(axis="columns") ##drop columns with any NA entries
        index_of_well = list(df.columns).index("Metadata_Well")
        nans = [] 
        for index, row in df.iterrows():
            entries = df.loc[index, :].values.tolist()
            letter_well = entries[index_of_well]
            barcode =  row["Metadata_Assay_Plate_Barcode"]
            platemap = row["Metadata_plate_map_name"]
            well_row = row_letter_to_index[letter_well[0]]
            well_col = letter_well[1:]
            numeric_well = "r" + well_row + "c" + well_col
            vector = np.array(entries[30:]) 
            key = platemap + "|" + barcode + "|r" + str(well_row) + "c" + str(well_col)
            if np.isnan(np.sum(vector)): 
                nans.append(key)
                continue 
            if key not in profile_map:
                profile_map[key] = vector 
            else:
                assert(np.array_equal(profile_map[key], vector))
    pickle.dump(profile_map, open("pickles/lincs/cellProfilerFeatures_from_{}.pkl".format(repo), "wb")) 

##Runner calls
directories = ["pickles/","pickles/JUMP1/", "pickles/lincs/", "csvs/", "csvs/JUMP1/", "csvs/JUMP1/learning/", "csvs/JUMP1/learning/moas/",  "csvs/lincs/", "csvs/lincs/learning/", "csvs/lincs/learning/moas"]
for directory in directories:
    if not os.path.isdir(directory):
        os.mkdir(directory)

# ##JUMP1
instantiateJumpDictionaries()
parseJumpImageNames()
getMOAMap()
createJumpImageCSV()
createJUMP1CompoundMOACSV()
createJUMP1CompoundMOACSVWellExcluded()
createJUMPCellAndTimeMap()
getLabelIndexMap("csvs/JUMP1/compound_images_with_MOA_no_polypharm.csv", study="JUMP1", target_column="moas")
getFullBarcodePlate()
createRowLetterConverters()

# ##LINCS
instantiateLINCSDictionaries()
parseLINCSImageName()
createLINCSImageCSV()
getLabelIndexMap("csvs/lincs/lincs_ten_micromolar_no_polypharm.csv", study="lincs", target_column="moas")

for study in ["JUMP1", "lincs"]:
    getPolyCompoundMOAs(study=study)
    createCompoundHoldout(study=study, min_compounds=2)
    getChannelStats(study=study)

##JUMP1 - profiles
parseJUMPCellProfilerData()
parseSpherizedJUMPCellProfilerData()
parseDeepProfilerFromModelOutput(study="JUMP1", method="median")

##lincs - profiles
parseLINCSCellProfilerData(drop_na="by column", repo="repo_level_3")
parseLINCSCellProfilerData(drop_na="by column", repo="repo_spherized")
parseDeepProfilerFromModelOutput(study="lincs", method="median")


