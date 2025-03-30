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
from langchain_core.tools import tool

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

# ================= Tooling =================

def FindDrug(drug_name: str):
    print("Finding Drug: ", )

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
    Finds the top k most similar drugs to the given drug_name based on cosine similarity.
    
    Args:
        drug_name (str): The name of the drug to compare.
        top_k (int): Number of similar drugs to return.

    Returns:
        List of tuples [(drug_name, similarity_score), ...]
    """
    # Fetch the target drug's embedding
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
    
    return list(cursor)

def PlotSmiles(smiles):
    """Generates and displays a 2D molecular structure from a SMILES string.
    
    Args:
        smiles (str): SMILES representation of the molecule.
    
    Returns:
        Boolean for if plotted or not
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.write(Draw.MolToImage(mol, size=(300, 300))) 
        return True # Generate 2D image
    else:
        raise False