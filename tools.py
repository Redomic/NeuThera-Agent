import os
import sys
import requests
import ast
import json
import hashlib
from datetime import datetime
from glob import glob
from io import StringIO

from db import db

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from arango import ArangoClient

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.llms.bedrock import Bedrock
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain.tools import Tool

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem

import streamlit as st

import boto3

# load_dotenv()
# arango_graph = ArangoGraph(db)
# bedrock_client = boto3.client(
#     "bedrock-runtime",
#     region_name=os.getenv("AWS_REGION"),
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
# )

# llm = Bedrock(client=bedrock_client, model_id="mistral.mistral-large-2402-v1:0")

# ================= Function =================

def FindDrug(drug_name: str):
    """
    Retrieves detailed information about a drug from the database.

    Args:
        drug_name (str): The name of the drug to search for.

    Returns:
        dict or None: A dictionary containing drug details if found, otherwise None.
    """

    query = """
    FOR d IN drug
        FILTER LOWER(d.drug_name) == LOWER(@name) OR LOWER(@name) IN TOKENS(d.synonym, "text_en")
        RETURN {
            _id: d._id,
            _key: d._key,
            accession: d.accession,
            drug_name: d.drug_name,
            cas: d.cas,
            unii: d.unii,
            synonym: d.synonym,
            key: d.key,
            chembl: d.chembl,
            smiles: d.smiles,
            inchi: d.inchi,
            generated: d.generated
        }
    """
    
    cursor = db.aql.execute(query, bind_vars={"name": drug_name})
    results = list(cursor)
    
    return results[0] if results else None

def FindSimilarDrugs(drug_name, top_k=5):
    """
    Finds the top k most similar drugs to the given drug based on cosine similarity.
    
    Args:
        drug_name (str): The name of the drug to compare.
        top_k (int): Number of similar drugs to return.

    Returns:
        List of tuples [(drug_name, similarity_score), ...]
    """

    query = f"""
        FOR d IN drug
            FILTER LOWER(d.drug_name) == LOWER(@drug_name) OR LOWER(@drug_name) IN TOKENS(d.synonym, "text_en")
            RETURN d.embedding
    """
    result = list(db.aql.execute(query, bind_vars={"drug_name": drug_name}))
    
    if not result:
        raise ValueError(f"Drug '{drug_name}' not found in the database.")
    
    embedding = result[0]

    aql_query = f"""
        LET query_vector = @query_vector
        FOR d IN drug
            FILTER LOWER(d.drug_name) != LOWER(@drug_name)
            LET score = COSINE_SIMILARITY(d.embedding, query_vector)
            SORT score DESC
            LIMIT @top_k
            RETURN {{ drug: d.drug_name, similarity_score: score }}
    """

    cursor = db.aql.execute(aql_query, bind_vars={"drug_name": drug_name, "query_vector": embedding, "top_k": top_k})
    results = list(cursor)

    if results:
        df = pd.DataFrame(results)
        st.table(df)
        return list(cursor)
    
    return []

def FindProteinsFromDrug(drug_name):
    """
    Finds all relevent proteins to the given drug

    Args:
        drug_name (str): The name of the drug.

    Returns:
        List[dict]: A list of PDB Ids
    """

    query = """
    FOR d IN drug 
        FILTER LOWER(d.drug_name) == LOWER(@drug_name)
        LIMIT 1  
        FOR v, e, p IN 1..2 OUTBOUND d._id
            GRAPH "NeuThera"
            FILTER IS_SAME_COLLECTION("protein", v)
            LIMIT 5
            RETURN DISTINCT { _key: v._key }
    """

    cursor = db.aql.execute(query, bind_vars={"drug_name": drug_name})
    return [doc["_key"] for doc in cursor]

def PlotSmiles2D(smiles):
    """Generates and displays a 2D molecular structure from a SMILES string.
    
    Args:
        smiles (str): SMILES representation of the molecule.
    
    Returns:
        Boolean - If plotted or not
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.write(Draw.MolToImage(mol, size=(300, 300))) 
        return True
    else:
        raise False

# ================= Tooling =================

find_drug_tool = Tool(
    name="FindDrug",
    func=FindDrug,
    description=FindDrug.__doc__
)

find_similar_drugs_tool = Tool(
    name="FindSimilarDrugs",
    func=FindSimilarDrugs,
    description=FindSimilarDrugs.__doc__
)

find_proteins_from_drug = Tool(
    name="FindProteinsFromDrug",
    func=FindProteinsFromDrug,
    description=FindProteinsFromDrug.__doc__
)

plot_smiles_2d_tool = Tool(
    name="PlotSmiles2D",
    func=PlotSmiles2D,
    description=PlotSmiles2D.__doc__
)

tools = [find_drug_tool, find_similar_drugs_tool, find_proteins_from_drug, plot_smiles_2d_tool]