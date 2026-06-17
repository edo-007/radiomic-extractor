from pathlib import Path
from pydantic import BaseModel, field_validator

class Paziente(BaseModel):
    nome: str
    path_ct: Path
    path_rt: Path

    @field_validator('path_ct', 'path_rt', mode="after")
    @classmethod
    def ensure_path_exist(cls, value: str):
        if not value.exists():
            raise ValueError("Il percorso specificato non esiste per CT/RTStruct")
        

class MirpExtractor(BaseModel):
    nome: str
    
    # def __init__(self, **settings):
    #     self.settings = settings
# 
    # def extract(paziente: Paziente):
    #     # Chiama direttamente mirp passando come parametri i settings
    #     pass

