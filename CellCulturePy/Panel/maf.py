### Author Douwe Spaanderman - 16 June 2020 ###

# This script reads all the files maf files and creates a major list of files

import pandas as pd
from pathlib import Path
import warnings
import numpy as np
import argparse
import time
import json

# Currently use sys to get other script - in Future use package
import os
import sys
path_main = ("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(path_main + '/Classes/')
sys.path.append(path_main + '/Utils/')
from gene_one_hot import one_hot
from help_functions import mean, str_to_bool, str_none_check

def maf_extract(maf):
    '''

    '''
    id_ = []
    data_dict= {}
    file_name = maf.stem.split('.')[0]
    i = 0
    with open(maf, 'r', encoding="latin-1") as f:
        try:
            for line in f:
                if line.startswith("#"):
                    continue
                elif not id_:
                    id_ = line.replace('\n', '').split('\t')
                else:
                    data_dict[i] = line.replace('\n', '').split('\t')
                    i += 1
        except:
            warnings.warn(f"File: {file_name}, had problems with unrecognizable symbols", DeprecationWarning)

    maf_frame = pd.DataFrame.from_dict(data_dict, orient="index", columns=id_)

    return maf_frame, file_name

#Filtering and frame transformation
def filter_frame(maf_row, cutoff=0, filter_protein_coding=False):
    '''

    '''
    #Change tumor_f to float and avg
    tumor_f = list(map(float, str(maf_row["tumor_f"]).split('|')))
    tumor_f = mean(tumor_f)
    
    maf_row["tumor_f"] = tumor_f
    
    makes_filters = True
    if tumor_f <= cutoff:
        makes_filters= False
    
    if filter_protein_coding == True:
        if maf_row["Protein_Change"] == '':
            makes_filters= False
        
    if makes_filters == True:
        return maf_row

def clean_frame(maf_frame):
    '''

    '''
    #CURRENTLY DANGEROUS AS I JUST TAKE TUMOR_SEQ_Allele2
    #Check allele, it is not actually phased so don't really know why it is called like this.
    #Also if tumor_f is 100 both alleles should have this mut which is not the case
    maf_frame["Tumor_Allele"] = maf_frame["Tumor_Seq_Allele2"]

    #dropna from filter_frame
    maf_frame = maf_frame.dropna()
    maf_frame = maf_frame[["Hugo_Symbol", 
                            "Entrez_Gene_Id", 
                            "Chromosome",
                            "Start_position",
                            "End_position",
                            "Variant_Classification",
                            "Variant_Type",
                            "Reference_Allele",
                            "Tumor_Allele",
                            "Matched_Norm_Sample_Barcode",
                            "Genome_Change",
                            "Annotation_Transcript",
                            "Transcript_Strand",
                            "Transcript_Exon",
                            "Transcript_Position",
                            "cDNA_Change",
                            "Codon_Change",
                            "Protein_Change",
                            "Ensembl_so_term",
                            "tumor_f"]]
    return maf_frame

def mutation_filter(data, classification_filter=True, ensembl_filter=False):
    '''
    mutation filter to filter out based on classification done in the maf file. Note that both filters generally use the same filtering, but
    I would advise to use default, because it is more standardized.

    input:
    

    '''
    if classification_filter == True:
        data = data[~data["Variant_Classification"].isin(["Intron", "lincRNA", "IGR", "5'Flank", "5'UTR", "Silent", "3'UTR", "RNA"])]
    
    if ensembl_filter == True:
        data = data[~data["Ensembl_so_term"].isin(["intron_variant", "intergenic_variant", "upstream_gene_variant", "5_prime_UTR_variant", "synonymous_variant", "3_prime_UTR_variant", ""])]  

    return data

#Create one-hot encoding for genes
def one_hot_encoder(data, all_genes:list, all_alterations:list):
    '''
    
    '''    
    # Initialize empty arrays
    gene_array = np.zeros(shape=(len(all_genes)))
    gene_alteration_1D_array = np.zeros(shape=(len(all_alterations)))
    
    # Get all keys
    gene = set(data["Hugo_Symbol"])
    gene_alt_1D = set(data["Hugo_Symbol"] + ":::" + data["Variant_Classification"])
    
    #Get index
    index_gene = [i for i, item in enumerate(all_genes) if item in gene]
    index_gene_alt_1D = [i for i, item in enumerate(all_alterations) if item in gene_alt_1D]
        
    #Change 0 -> 1 for index
    np.put(gene_array, index_gene, 1)
    np.put(gene_alteration_1D_array, index_gene_alt_1D, 1)
    
    #Create class
    flat = one_hot(gene_array, all_genes)
    all_alt = one_hot(gene_alteration_1D_array, all_alterations)
    
    #2D array
    all_alt_2D = all_alt.make_2D(int(len(all_alterations)/len(all_genes)))
    
    # Sanity checks
    flat.sanity()
    all_alt.sanity()
    all_alt_2D.sanity()

    #Create classes
    return flat, all_alt, all_alt_2D

#Create one-hot encoding for empty failed mutation files
def one_hot_empty(all_genes:list, all_alterations:list):
    '''
    
    '''    
    # Initialize empty arrays
    gene_array = np.zeros(shape=(len(all_genes)))
    gene_alteration_1D_array = np.zeros(shape=(len(all_alterations)))
    
    #Create class
    flat = one_hot(gene_array, all_genes)
    all_alt = one_hot(gene_alteration_1D_array, all_alterations)
    
    #2D array
    all_alt_2D = all_alt.make_2D(int(len(all_alterations)/len(all_genes)))
    
    # Sanity checks
    flat.sanity()
    all_alt.sanity()
    all_alt_2D.sanity()

    #Create classes
    return flat, all_alt, all_alt_2D

def main_maf(directory, filter_protein_coding=False, classification_filter=True, ensembl_filter=False, Save=False, Show=True):
    '''

    '''
    if directory.endswith("/"):
        pathlist = Path(directory).glob('*.maf*')
        number_of_files = len(list(pathlist))
        pathlist = Path(directory).glob('*.maf*')
    else:
        pathlist = Path(directory).glob('/*.maf*')
        number_of_files = len(list(pathlist))
        pathlist = Path(directory).glob('/*.maf*')
    
    data = []
    cache_failed = []
    for idx, path in enumerate(pathlist):
        maf_frame, file_name = maf_extract(path)

        #print("now doing: {}".format(file_name))
        maf_frame = maf_frame.apply(filter_frame, cutoff=0, filter_protein_coding=filter_protein_coding, axis=1)

        if type(maf_frame) == pd.core.series.Series or maf_frame.empty == True:
            print(f"{file_name} has no mutations that made tumor fraction cutoff or in protein coding")
            cache_failed.append(file_name)
            continue
    
        maf_frame = clean_frame(maf_frame)
        maf_frame["file"] = file_name
        
        data.append(maf_frame)
        if idx % 10 == 0:
            print(f'done {idx+1} out of {number_of_files}')

    data = pd.concat(data)
    data = mutation_filter(data, classification_filter=classification_filter, ensembl_filter=classification_filter)
    
    # Now create one-hot
    data_summary = []
    unique_names = data["file"].unique()
    all_genes = data["Hugo_Symbol"].unique()
    #Remove Unknown
    all_genes = all_genes[all_genes != "Unknown"]
    all_alterations = data["Variant_Classification"].unique()
    all_alterations = [str(x) + ":::" + str(y) for x in all_genes for y in all_alterations]
    for i, name in enumerate(unique_names):
        tmp_data = data[data["file"] == name]
        flat, all_alt, all_alt_2D = one_hot_encoder(tmp_data, all_genes, all_alterations)
        
        # Create dataframe
        tmp_data = pd.DataFrame({
                        "File": name,
                        "Flat_one_hot": flat,
                        "Alt_one_hot": all_alt,
                        "Alt_2D": all_alt_2D
        }, index=[0])
        
        
        data_summary.append(tmp_data)
            
        if i % 10 == 0:
            print(f'done {i+1} out of {len(unique_names)}')

    if not cache_failed:
        print("No failed mutation files")
    else:
        for i, name in enumerate(cache_failed):
            flat, all_alt, all_alt_2D = one_hot_empty(all_genes, all_alterations)
            # Create dataframe
            tmp_data = pd.DataFrame({
                            "File": name,
                            "Flat_one_hot": flat,
                            "Alt_one_hot": all_alt,
                            "Alt_2D": all_alt_2D
            }, index=[0])
            
            data_summary.append(tmp_data)

            if i % 10 == 0:
                print(f'done {i+1} out of {len(unique_names)}')

    data_summary = pd.concat(data_summary)
    print(data_summary)

    if Save != False:
        Save_cache = Save
        if not Save.endswith(".pkl"):
            Save = Save + "maf_extract.pkl"
            
        Save_summary = "/".join(["_summary.".join(x.split(".")) if i+1 == len(Save.split("/")) else x for i, x in enumerate(Save.split("/"))])
        data.to_pickle(Save)
        data_summary.to_pickle(Save_summary)
            
        with open(Save_cache + "Cache/" + "cache_failed_maf.json", 'w') as f:
            json.dump(cache_failed, f, indent=2) 

    if Show == True:
        print(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read and combine all maf data with the data")
    parser.add_argument("Path", help="path to directory with maf files")
    parser.add_argument("-s", dest="Save", nargs='?', default=False, help="location of file")
    parser.add_argument("-d", dest="Show", nargs='?', default=True, help="Do you want to show the plot?")
    parser.add_argument("-p", dest="filter_protein_coding", nargs='?', default=True, help="Do you want to show the plot?")
    parser.add_argument("-c", dest="classification_filter", nargs='?', default=False, help="Do you want to show the plot?")
    parser.add_argument("-e", dest="ensembl_filter", nargs='?', default=False, help="Do you want to show the plot?")

    args = parser.parse_args()
    start = time.time()
    main_maf(directory=args.Path, filter_protein_coding=str_to_bool(args.filter_protein_coding), classification_filter=str_to_bool(args.classification_filter), ensembl_filter=str_to_bool(args.ensembl_filter), Save=args.Save, Show=str_to_bool(args.Show))
    end = time.time()
    print('completed in {} seconds'.format(end-start))