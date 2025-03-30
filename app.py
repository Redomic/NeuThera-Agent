import os
import sys
import requests
import ast
import json
import hashlib
from datetime import datetime
from glob import glob
from io import StringIO
import streamlit as st
from tools import FindDrug
import py3Dmol

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
from streamlit.components.v1 import html
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw, AllChem
from tools import FindDrug

import boto3

# ================= Application =================

load_dotenv()
arango_graph = ArangoGraph(db)
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

llm = Bedrock(client=bedrock_client, model_id="mistral.mistral-large-2402-v1:0")

from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import BedrockChat

# llm = BedrockChat(model_id="mistral.mistral-large-2402-v1:0")


# def text_to_aql(query: str):
#     """Execute a Natural Language Query in ArangoDB, and return the result as text."""
    
#     # llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

#     chain = ArangoGraphQAChain.from_llm(
#         llm=llm,
#         graph=arango_graph,  # Assuming arango_graph is already initialized
#         verbose=True,
#         allow_dangerous_requests=True
#     )
    
#     result = chain.invoke(query)

#     return str(result["result"])

# Define text_to_aql as a tool
# text_to_aql_tool = Tool(
#     name="text_to_aql",
#     func=text_to_aql,  # Your function
#     description="Execute a Natural Language Query in ArangoDB, and return the result as text"
# )

# # Initialize the agent
# agent = initialize_agent(
#     tools=[text_to_aql_tool],  # Add tools here
#     llm=llm,
#     agent="zero-shot-react-description",  # Can be different depending on implementation
#     verbose=True
# )

# # Run agent with a query
# response = agent.run("Find all drugs targeting BRCA1.")

# print(response)
def plot_2d_smiles(smiles):
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

import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem

def plot_3d(smiles):
    """Generates an interactive 3D molecular structure from a SMILES string.
    
    Args:
        smiles (str): SMILES representation of the molecule.
    
    Returns:
        plotly.graph_objects.Figure or None: 3D visualization of the molecular structure.
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES structure!")

    mol = Chem.AddHs(mol)  # Add hydrogen atoms

    # Try to generate 3D coordinates
    status = AllChem.EmbedMolecule(mol, maxAttempts=100) 
    if status == -1:  # If embedding fails
        raise ValueError("Failed to generate 3D coordinates for this molecule.")

    # Optimize the molecule
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        raise ValueError("Molecular optimization failed.")

    # Extract atomic coordinates
    conformer = mol.GetConformer()
    if not conformer.Is3D():
        raise ValueError("Conformer is not in 3D.")

    atom_positions = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    # Create 3D scatter plot for atoms
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=atom_positions[:, 0], y=atom_positions[:, 1], z=atom_positions[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='blue', opacity=0.8),
        text=atom_symbols,
        textposition="top center"
    ))

    # Add bonds as lines
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        fig.add_trace(go.Scatter3d(
            x=[atom_positions[start][0], atom_positions[end][0]],
            y=[atom_positions[start][1], atom_positions[end][1]],
            z=[atom_positions[start][2], atom_positions[end][2]],
            mode='lines',
            line=dict(color='gray', width=3)
        ))

    # Format layout
    fig.update_layout(
        title="3D Molecular Structure",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=600, height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    

    return st.write(fig)



find_drug_tool=Tool(
        name="FindDrug",
        func=FindDrug ,
        description="Find the  details for the given drug ",
    )
plot_smiles=Tool(
    name="Plot2dSmiles",
    func=plot_2d_smiles,
    description="PLot the smiles from the given drug related smiles "
)
plot_3d=Tool(
    name="Plot3dSmiles",
    func=plot_3d,
    description="Plot the 3d smiles from the given drug related smiles"
)   
agent=initialize_agent(
        tools=[find_drug_tool,plot_smiles,plot_3d],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True

    )
    

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Type your drug-related query...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run agent dynamically
    result = agent.run(user_input)

    # Display AI response       
    with st.chat_message("assistant"):
        st.markdown(f"🤖 {result}")
    st.session_state.messages.append({"role": "assistant", "content": result})

