import streamlit as st
from streamlit_chat import message

from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
          return super().find_class(module, name)


class LLM:

  corpus_sentence_path = "./corpus_sentences_3.pkl"
  corpus_embeddings_path = "./corpus_embeddings_3.pkl"

  def __init__(self):
    
    self.embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    file1 = open(self.corpus_sentence_path,'rb')
    self.final_corpus = pickle.load(file1)
    file1.close()

    file2 = open(self.corpus_embeddings_path,'rb')

    self.corpus_embeddings = CPU_Unpickler(file2).load()
    file2.close()

    if isinstance(self.corpus_embeddings, torch.Tensor):
        self.corpus_embeddings = self.corpus_embeddings.to('cpu')

  
  def get_response(self, query):
    # top_k = min(5, len(self.final_corpus))

    query_embedding = self.embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
    l =  [(i,j,k) for i,j,k in zip(cos_scores, self.corpus_embeddings, self.final_corpus) if i>0.5]
    l = sorted(l, key = lambda x:x[0], reverse=True)
    top_results = l[:5]

    return [result[2] for result in top_results]

obj = LLM()
import streamlit as st
import random
import time

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How are you feeling?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = obj.get_response(prompt)[0]
        print(full_response)
        # assistant_response = random.choice(
        #     [
        #         "Hello there! How can I assist you today?",
        #         "Hi, human! Is there anything I can help you with?",
        #         "Do you need help?",
        #     ]
        # )
        # # Simulate stream of response with milliseconds delay
        # for chunk in assistant_response.split():
        #     full_response += chunk + " "
        #     time.sleep(0.05)
        #     # Add a blinking cursor to simulate typing
        #     message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
