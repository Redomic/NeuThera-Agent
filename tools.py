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

import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem

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

# ================= Functions =================

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

def FindSimilarDrugs(drug_name):
    """
    Finds the top k most similar drugs to the given drug based on cosine similarity.
    
    Args:
        drug_name (str): The name of the drug to compare.

    Returns:
        List of tuples [(drug_name, similarity_score), ...]
    """
    
    top_k = 5

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
        return results
    
    return results

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

def PlotSmiles3D(smiles):
    """Generates an interactive 3D molecular structure from a SMILES string.
    
    Args:
        smiles (str): SMILES representation of the molecule.
    
    Returns:
        boolean for if plotted or not 
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)  # Add hydrogen atoms

    # Try to generate 3D coordinates
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if status == -1:  # If embedding fails
        return False

    # Optimize the molecule
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return False

    # Extract atomic coordinates
    conformer = mol.GetConformer()
    if not conformer.Is3D():
        return False

    atom_positions = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    # Define colors: Green for O, Red for H, Blue for others
    atom_colors = ['green' if atom == 'O' else 'red' if atom == 'H' else 'blue' for atom in atom_symbols]

    # Create 3D scatter plot for atoms
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=atom_positions[:, 0], y=atom_positions[:, 1], z=atom_positions[:, 2],
        mode='markers+text',
        marker=dict(size=3, color=atom_colors, opacity=0.8),
        text=atom_symbols,
        textposition="top center",
        showlegend=False  # Hide the legend
    ))

    # Add bonds as lines
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        fig.add_trace(go.Scatter3d(
            x=[atom_positions[start][0], atom_positions[end][0]],
            y=[atom_positions[start][1], atom_positions[end][1]],
            z=[atom_positions[start][2], atom_positions[end][2]],
            mode='lines',
            line=dict(color='gray', width=3),
            showlegend=False  # Hide legend for bonds
        ))

    # Format layout
    fig.update_layout(
        title="3D Molecular Structure",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        width=600, height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False  # Hide legend globally
    )
    if fig:
        st.write(fig)
        return True
    else:
        return False    

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

find_proteins_from_drug_tool = Tool(
    name="FindProteinsFromDrug",
    func=FindProteinsFromDrug,
    description=FindProteinsFromDrug.__doc__
)

plot_smiles_2d_tool = Tool(
    name="PlotSmiles2D",
    func=PlotSmiles2D,
    description=PlotSmiles2D.__doc__
)

plot_smiles_3d_tool = Tool(
    name="PlotSmiles3D",
    func=PlotSmiles3D,
    description=PlotSmiles3D.__doc__
)

tools = [find_drug_tool, find_similar_drugs_tool, find_proteins_from_drug_tool, plot_smiles_2d_tool, plot_smiles_3d_tool]