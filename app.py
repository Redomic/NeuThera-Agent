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