import os

import streamlit as st

from tools import tools 
from db import db

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.llms import Bedrock
from langchain_community.graphs import ArangoGraph
from langchain.agents import initialize_agent
from langchain.callbacks.base import BaseCallbackHandler

import boto3

# ================= Application =================

load_dotenv()

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

llm = Bedrock(client=bedrock_client, model_id="mistral.mistral-large-2402-v1:0")

agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True,
    )

agent.run("Give me details for the drug goserelin")