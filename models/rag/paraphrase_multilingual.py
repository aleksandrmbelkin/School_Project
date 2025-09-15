from sentence_transformers import SentenceTransformer
import numpy as np
import re
from pathlib import Path
from typing import List, Dict
import os

if not os.path.isdir('./models/rag/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'):
    print("Загружаем модель paraphrase-multilingual-MiniLM-L12-v2")
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name, cache_folder='./models/rag')
    print("Модель paraphrase-multilingual-MiniLM-L12-v2 загружена")