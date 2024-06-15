import os 
from dotenv import load_dotenv  
import streamlit as st 
from PyPDF2 import PdfReader 

from langchain_text_splitters import RecursiveCharacterTextSplitter  # < -- for spliting text 
from langchain_google_genai import GoogleGenerativeAIEmbeddings          # <-- embedding  
import google.generativeai as genai  

from langchain.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate 



load_dotenv() 

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))



#1) now we read  all the pdf and extract all text and save it in variables 

def read_pdf_text(pdf_docs):
    text="" 
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text() 
    
    return text  


# 2) Now we get text we divide these text into chunks size  ; 

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    
    return chunks
    

# 3) we have the chunks and now we convert it into vectors and store\save it in to Faiss database in local or remote 

def get_vector_store(text_chunks) :
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")     #<-- it  model present init 
    vector_store=FAISS.from_texts(text_chunks , embedding=embeddings) 
    vector_store.save_local("faiss_index")                                     #< -- saving it in local 
    
    
  
 # 4) define model , prompt , chain  
 
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not inprovided context just say, "answer is not available in the context", 
    don't provide the wrong answer\n\n
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """    
    
    
    model=ChatGoogleGenerativeAI(model="gemini-pro" , temperature=0.4)  

    prompt=PromptTemplate(template=prompt_template , input_variables=["context","question"]) 
    
    chain=load_qa_chain(model, prompt=prompt ,chain_type="stuff" )  # stuff inside it will do text summerizaiton 
    
    return chain 





# 5) define user interface  


def user_input(user_question):
    embeddings =GoogleGenerativeAIEmbeddings(model = "models/embedding-001") 
    
    Local_faiss_db =FAISS.load_local("faiss_index" , embeddings) 
    
    docs=  Local_faiss_db.similarity_search(user_question) 
    
    chain= get_conversational_chain()     # 4 
    
    response= chain( 
        {"input_document" :docs , "question":user_question}
        ,return_only_outputs=True) 
    
    
    print(response)
    st.write("Reply is : ", response["output_text"])
    
    
    
    
    
    
# Create stremlit app for multile Pdf  




def main():
    st.set_page_config("Chat with PDF")
    st.header("Interact with Your PDFs Like Never Before with Gemini 💡📄")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)    #5 

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = read_pdf_text(pdf_docs)   #1 
                text_chunks = get_text_chunks(raw_text) # 2
                get_vector_store(text_chunks)  # 3 
                st.success("Done")



if __name__ == "__main__":
    main()
