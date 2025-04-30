from dotenv import load_dotenv
import os

load_dotenv()  

#!pip install -q cassio langchain openai streamlit
#!pip install -U langchain-community
#!pip install PyPDF2

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display
from PyPDF2 import PdfReader
import cassio

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.environ.get("ASTRA_DB_ID")

uploader = widgets.FileUpload(
    accept='.pdf',
    multiple=False
)
display(uploader)

pdfreader = None

def handle_upload(change):
    global pdfreader
    for filename, fileinfo in uploader.value.items():
        with open(filename, 'wb') as f:
            f.write(fileinfo['content'])
        print(f"Uploaded file: {filename}")
        pdfreader = PdfReader(filename)
        print(f"Number of pages: {len(pdfreader.pages)}")

uploader.observe(handle_upload, names='value')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

print(raw_text[:500])

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
#llm = OpenAI(openai_api_key=OPENAI_API_KEY)
#embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

astra_vector_store.add_texts(texts[:50])

print("Inserted %i headlines." % len(texts[:50]))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarizer = pipeline("summarization", model="t5-small")

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False
    print("\nQUESTION: \"%s\"" % query_text)
    results = astra_vector_store.similarity_search_with_score(query_text, k=4)

    if results:
        top_doc, top_score = results[0]
        summary = summarizer(top_doc.page_content, max_length=100, min_length=30, do_sample=False)
        print("ANSWER (summarized): \"%s\"\n" % summary[0]['summary_text'])
    else:
        print("No documents found.\n")

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in results:
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
