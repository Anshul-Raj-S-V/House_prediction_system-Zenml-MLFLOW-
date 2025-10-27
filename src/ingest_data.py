import os

import zipfile
import pandas as pd
from abc import ABC , abstractmethod


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self,file_path: str)-> pd.DataFrame:

        if not file_path.endswith(".zip"):
            raise ValueError("It is not a zip file")
        

        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall("extracted_data")

            extracted_files = os.listdir("extracted_data")
            csv_files = [f for f in extracted_files if f.endswith(".csv") ]


            if len(csv_files)==0:
                raise FileNotFoundError("No CSV FILES")
            if len(csv_files)>1:
                raise ValueError("Multiple CSV file , Specify which one to use")
            

        csv_file_path = os.path.join("extracted_data",csv_files[0])
        df = pd.read_csv(csv_file_path)




        return df
    



class DataIngestorFactory:
    @staticmethod

    def get_data_ingestor(file_extension: str) -> DataIngestor:

        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor avialbe for file extension: {file_extension}")
        
if __name__ == "__main__":
    factory = DataIngestorFactory()
    ingestor = factory.get_data_ingestor(".zip")
    df = ingestor.ingest("data/archive.zip")  # replace with your zip filename
    print(df.head())
