import pandas as pd
import os


def stem_name(path: str) -> str:
    """:return: basename without extension"""
    base_name = os.path.basename(path)
    return base_name[: base_name.rfind('.')]


def pdf2csv(pdf_path: str, output_dir='') -> NotImplemented:
    df = pd.DataFrame(
        {
            'paragraph':
                (
                    'pdf2csv has not been implemented yet, this is a sample csv',
                    'pdf2csv 尚未被實作，這是範例csv',
                    'today our teams around the world infuse apples deeply held values into everything we make that work can take many forms but whether we are protecting the right to privacy designing technology that is accessible to all or using more recycled material in our products than ever we are always working to make a difference for the people we serve and the planet we inhabit'
                )
        }
    )
    csv_path = os.path.join(output_dir, stem_name(pdf_path) + '.csv')
    df.to_csv(csv_path, sep=',', index=False, encoding='utf-8')
