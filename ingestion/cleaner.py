"""
👨‍💻 Person 2: Text Cleaner
Responsibility: Clean the extracted text.

CRITICAL REQUIREMENT:
Takes the standard list of dictionaries from the loader and returns a 
cleaned version of the SAME format.
[
  {
    "text": "Cleaned text...",
    "source": "filename.pdf",
    "page": 1
  }
]

INNOVATION OPPORTUNITIES:
- Remove redundant whitespaces, special hidden characters.
- Fix broken hyphenations (e.g., "demo- \nstration" -> "demonstration").
- Remove standard headers/footers or boilerplate text.
"""

def clean_text(docs: list[dict]) -> list[dict]:
    """
    Cleans the text content of the documents.
    
    Args:
        docs (list[dict]): The output from loader.py
        
    Returns:
        list[dict]: Cleaned documents, preserving 'source' and 'page'.
    """
    cleaned_docs = []
    
    # TODO: Implement cleaning logic here.
    # 1. Iterate through docs
    # 2. Apply regex or string replacement on doc['text']
    # 3. Add to cleaned_docs
    
    # Example snippet:
    # for d in docs:
    #     clean_tgt = d['text'].strip().replace('\n\n', '\n')
    #     cleaned_docs.append({"text": clean_tgt, "source": d['source'], "page": d['page']})
    
    return cleaned_docs
