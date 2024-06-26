from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
docs = PyPDFLoader("aiact_final_draft.pdf").load_and_split(text_splitter = text_splitter ) 


doc_contents = [doc.page_content for doc in docs]


# List to hold the IDs for each document
doc_ids = list(range(1, len(docs) + 1))


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device='cuda')
vectors = model.encode(doc_contents)


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import streamlit as st


client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name="cf_data",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

client.upload_collection(
    collection_name="cf_data",
    ids=[i for i in range(len(doc_contents))],
    vectors=vectors,
    
)


from dspy.retrieve.qdrant_rm import QdrantRM
qdrant_retriever_model = QdrantRM("cf_data", client, k=3)

import dspy
lm = dspy.OllamaLocal(model="llama3",timeout_s = 180)


dspy.settings.configure(rm=qdrant_retriever_model, lm=lm)


class GenerateAnswer(dspy.Signature):
    """Answer questions with logical factoid answers."""

    context = dspy.InputField(desc="will contain an AI act related document")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="a answer within 20 to 30 words")


def get_context(text):
    query_vector = model.encode(text)


    hits = client.search(
        collection_name="cf_data",
        query_vector=query_vector,
        limit=3  # Return 5 closest points
    )
    s=''
    for x in [doc_contents[i.id] for i in hits]:
        s = s + x
    return s

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()


        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)


    def forward(self, question):
        context = get_context(question)
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
rag = RAG()
def respond(query):
    response = rag(query)
    return response.answer



st.set_page_config(page_title="DSPy RAG Chatbot", page_icon=":robot_face:")


st.markdown("""
<div style="text-align: center;">
            <img src="https://dspy-docs.vercel.app/img/logo.png" alt="Chatbot Logo" width="100"/>
    <img src="https://img.freepik.com/premium-vector/robot-icon-chat-bot-sign-support-service-concept-chatbot-character-flat-style_41737-796.jpg?" alt="Chatbot Logo" width="200"/>
    <h1 style="color: #0078D7;">DSPy based RAG Chatbot</h1>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<p style="text-align: center; font-size: 18px; color: #555;">
    Hello! Just ask me anything from the dataset.
</p>
""", unsafe_allow_html=True)


st.markdown("<hr/>", unsafe_allow_html=True)

user_query = st.text_input("Enter your question:", placeholder="E.g., What is the aim of AI act?")

if st.button("Answer"):
    bot_response = respond(user_query)
   
    st.markdown(f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <h4 style="color: #0078D7;">Bot's Response:</h4>
        <p style="color: #333;">{bot_response}</p>
    </div>
    """, unsafe_allow_html=True)