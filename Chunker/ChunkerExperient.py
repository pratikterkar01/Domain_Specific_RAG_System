import os

from Utils import Config
from Chunker import Ingestion,StructureAwareChunking,ParentChildChunking,ContextualTokenAwareChunking
from Embeding.Embeding import Embeding
from Embeding.Embeding import Embeding

global_Variable = Config.get_settings()

dbRoot = global_Variable.DB_ROOT
def pdf_ingection(pdf_Root):
    documen_parse = Ingestion.Ingestion.parse_pdf(pdf_Root)

    ## chunking stratagy
    ## 1. Structure Aware
    structureAwareChunking = StructureAwareChunking.StructureAwareChunking.structure_aware_chunking(documen_parse)


    ##Initialize embeding model
    embedinngs = Embeding()
    bge_m3 = embedinngs.embed(model_name="BAAI/bge-m3")

    bge_m3_structureAwareChunking = embedinngs.vectore_store(bge_m3,structureAwareChunking)


pdf_file_name  = ["D:\\AiEnergyAssitanceForBuilding\\PdfResourc\\Used\\ECBC_2024.pdf","D:\\AiEnergyAssitanceForBuilding\\PdfResourc\\Used\\ECBC_HVAC_TIP_SHEET_2011.pdf","D:\\AiEnergyAssitanceForBuilding\\PdfResourc\\Used\\InputOutputReference.pdf"]

documen_parse = Ingestion.Ingestion.parse_pdf(pdf_file_name[2])

