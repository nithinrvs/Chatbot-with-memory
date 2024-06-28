import streamlit as st
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=4000)

if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = ConversationChain(llm=model, memory=st.session_state.memory, verbose=False)

def main():
    st.title("Conversational AI with Memory")
    st.write("Talk to the AI:")
    
    chat_area = st.empty()

    with st.form(key='user_input_form'):
        user_input = st.text_input("You: ", key="user_input")
        submit_button = st.form_submit_button(label='Enter')

    if submit_button and user_input:
        with chat_area.container():
            st.write(f"User input: {user_input}")
            st.write("Conversation chain:")
            response = st.session_state.conversation_chain.predict(input=user_input)
            st.write(response)
        
        # Clear the input box
        st.session_state.user_input = ""

if __name__ == "__main__":
    main()
