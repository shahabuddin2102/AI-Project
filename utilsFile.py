from io import BytesIO
import pandas as pd
import PyPDF2
import base64
import docx

def extract_text_from_file(file_content: bytes, file_extension: str) -> str:
    try:
        file_extension = file_extension.lower()
        if file_extension == "pdf":
            reader = PyPDF2.PdfReader(BytesIO(file_content))
            return "".join(page.extract_text() for page in reader.pages)
        elif file_extension == "docx":
            doc = docx.Document(BytesIO(file_content))
            return "\n".join(p.text for p in doc.paragraphs)
        elif file_extension == "csv":
            df = pd.read_csv(BytesIO(file_content))
            return "\n".join(" ".join(map(str, row)) for _, row in df.iterrows())
        elif file_extension == "txt":
            return file_content.decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        raise Exception(f"Text extraction error: {str(e)}")