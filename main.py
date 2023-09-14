# Use a pipeline as a high-level helper
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
#from langchain import PromptTemplate, LLMChain, OpenAI
# import os
# import requests


load_dotenv(find_dotenv())

text = "Sami est un bon mec à Societe Generale"
def ner(text):
    local_path = "./local_model/bert-base-NER/"
    model="dslim/bert-base-NER"
    pipe = pipeline("token-classification", model=local_path, aggregation_strategy="average") 
    #pipe = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
    entity_recognition = pipe(text)
    
    print(entity_recognition)
    return entity_recognition



result = ner(text)



