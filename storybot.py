# Note: I am using the plugin 'Better Comments' for easier comment legibility. This is why some comments might have additional asterisks, exclamation marks, etc.
# I can highly recommend the plugin if you are working in VS Code (chances are high you are already using it..)

### * 01 FINAL VARIABLES * ###
# This section contains all final variables referenced at a later stage in the program.

setup_link = "https://platform.openai.com/docs/quickstart/account-setup"

# To be able to initialize the session state variables in a loop, this list references all keys to be initialized.
sessionStateKeys = ['logged_in', 'toast_msg', 'conversation_stage']

# The app allows the user to choose the GPT model it should use for generating the stories. The options, however, are limited to these three models.
gpt_model_options = ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',]

# TODO add explainer
chatInputPrompts = [
    'Start off your journey by entering the first few lines of a story, or give me a general theme.',
    'How should the story continue? You can provide your own input or choose from the suggestions above. Say END STORY to end the story.',
]

# The initial message from the AI chatbot is hardcoded to waste as little tokens as possible. To not make the experience too boring for the user, the chat will pick one of these variations at random.
aiIntroMessages = [
    "Hey! I'm Bedtime Buddy, your bedtime storyteller. Do you want to hear a new adventure? Just tell me a few things you like, and I'll start a story for you!",
    "Hi there! I'm Bedtime Buddy, here to tell you a bedtime story. Want to go on an adventure? Just give me some ideas, and I'll begin the tale!",
    "Hello, friend! I'm Bedtime Buddy, your storyteller for tonight. Ready for a new story? Let me know what you're interested in, and I'll create a magical beginning!",
    "Hey, buddy! I'm Bedtime Buddy, and I've got a story for you. Want to hear it? Just share a few things you enjoy, and I'll start the adventure!",
    "Hiya! I'm Bedtime Buddy, your bedtime storyteller extraordinaire. Ready for a new adventure? Tell me what you're into, and I'll weave a tale just for you!",
    "Hey, sleepyhead! It's me, Bedtime Buddy, here to tell you a story. Want to hear one? Give me some ideas, and I'll begin the adventure!"
]

### * 02 IMPORTS * ###
# All these imports are needed to either display the app correctly, make requests to the API, or to work with OpenAI.

import random
import numpy as np
import streamlit as st # Streamlit App functionality
import requests # API Requests
from time import sleep
# from langchain_openai import ChatOpenAI
from openai import OpenAI

### * CLASSES * ###
# These classes define 'bespoke' objects and their attributes, e.g., what a message is with respect to this program. In any other environment beside this WMCC class,
# I would advise to refactor this in a different file, but due to the limitations of the submission to deliver everything in one file, I have to keep the classes here.

class Message:
    def __init__(self, role, avatar, message):
        self.role = role
        self.avatar = avatar
        self.message = message


### * 03 FUNCTIONS * ###
# Helper functions and handling of chat requests, summarizing the contents, etc.

# Function to check the validity of the OpenAI API key
def check_api_key(api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://api.openai.com/v1/engines"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return True, "Valid API Key"
    elif response.status_code == 401:
        return False, "Invalid API Key: Unauthorized"
    else:
        return False, f"Failed with status code: {response.status_code}"


# Helper Function that takes text input and uses it to display a toast widget in streamlit. Unless otherwise specified, a checkbox icon will be provided automatically.
def display_toast_msg(msg, icon='✅'):
    st.toast(msg, icon=icon)
    st.session_state.toast_msg = False
    return

# TODO Add Explainer Comment
def showChatHistory():
    if 'chat_history' in st.session_state:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg.role, avatar=msg.avatar):
                st.markdown(msg.message)

# TODO Add Explainer Comment
def addMessage(msg):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append(msg)
    else:
        st.session_state.chat_history.append(msg)

# TODO Add Explainer Comment
def generateMessages(prompt):
    prev_messages=[]
    if 'chat_history' in st.session_state:
        for msg in st.session_state.chat_history[1:]:
            m={"role": msg.role, "content": msg.message}
            prev_messages.append(m)
    u={"role": "user", "content": prompt}
    prev_messages.append(u)
    return prev_messages

# TODO Add Explainer Comment
def getBotResponse(client,model,prompt): 
    with st.chat_message("assistant", avatar="🤖"):
        stream = client.chat.completions.create(
            model = model,
            messages = generateMessages(prompt),
            stream = True
        )
        return st.write_stream(stream)


# Returns a hard-coded greeting message for the user. Using the OpenAI interface for this would be inefficient, since the text is not helpful for generating the messages.
def introMessage():
    addMessage(Message("assistant", "🤖", aiIntroMessages[random.randint(0,len(aiIntroMessages)-1)]))


# Initializes the chatbot with a system message, telling the bot what to do and how to behave.
def initSystemMessage():
    addMessage(Message("system", "s","You are a chatbot whose sole purpose is to write bedtime stories for childen. Your output must at all times be child-friendly, easy to understand for a young audience, and exciting to read. It should not contain difficult words and should be written in a bedtime story style. Your answers should be structured in a JSON format, and include the following keys: answer, dalle-prompt, option1, option2, option3. answer is your story fragment, which must contain a small cliffhanger at the end. Only include the story continuation, no boilerplate answer. The story must relate to the story parts previously generated during the conversation and must not contain any plotholes. dalle-prompt is a shorter prompt summarizing the current story part for image creation. the option keys should contain keywords of 1 to 3 words, suggesting how the story should continue.", ))


### * 04 MAIN FUNCTION * ###
# Where all the frontend layouting happens. This is the core of my streamlit application.

def main():

    # Set the streamlit layout to 'wide' to get more space, and make sure the sidebar is visible per default.
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    # Streamlit Sidebar - if the user is not logged in, not much will be displayed here. As soon as the user is logged in, they will have the option to log out again.
    st.sidebar.title("Login Status")

    # Initialize Session state variables at the first run of the script (when they have not been initialized yet)
    for key in sessionStateKeys:
        if key not in st.session_state:
            st.session_state[key] = False

    # Checks every rerun if a toast message has been stored prior to executing the rerun, and shows the toast if necessary.
    if st.session_state.toast_msg != False:
        display_toast_msg(st.session_state.toast_msg)

    # * LOGIN / WELCOME PAGE if the user is not logged in / has just logged out
    if not st.session_state.logged_in:

        st.sidebar.info("You are not logged in. Please provide a valid OpenAI API key to continue.")
        
        st.title("Welcome to BedtimeBuddy 🦄")
        st.subheader("Your Nightly Dose of ✨ Dream Dust! ✨")
        st.write("Attention Parents: To start diving into your next adventure bedtime story with your kids, please provide a valid API key for OpenAI.")
        st.info("First time loggin in, and not sure how to get an API key? [Check out the OpenAI Documentation!](%s)" % setup_link )

        # Ask for the API key of the user, which will then temporarily be stored in the session state for the session's duration.
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")

        # Button that handles the login request of the user, using the check_api_key function to check the validity of the key,
        # before moving on to the logged-in state if the key is actually valid.

        if st.button("Login"):
            isValid, message = check_api_key(api_key)
            if isValid:
                st.session_state.logged_in = True
                st.session_state.api_key = api_key
                st.session_state.toast_msg = "Logged in successfully. Let's create a great story together!"
                st.session_state.conv_stage = 0
                st.rerun()
            else:
                st.error(f"Whoops, that didn't work. Have you checked if the key is correct? (Error: {message})")

    # * LOGGED-IN DASHBOARD / MAIN SCREEN if the user has logged in
    else:

        st.sidebar.success("You are logged in.")
        st.sidebar.write("Feeling sleepy? You can log out of the system to prevent unauthorized use. Your API key will not be stored permanently in this program.")
        logout = st.sidebar.button("Logout")
        st.sidebar.divider()
        st.sidebar.subheader("GPT Model Selector")
        st.sidebar.write("You can choose the GPT model  used for generating the bedtime stories.")
        gpt_model_selector = st.sidebar.selectbox(
            'GPT-model in use:',
            gpt_model_options,
            index=0,
            help="GPT-4 models typically yield better results, but at a higher cost.",
        )

        if "gpt_model" not in st.session_state:
            st.session_state.gpt_model = gpt_model_selector
            
        st.sidebar.divider()
        st.sidebar.subheader("Reset Conversation")
        st.sidebar.write("Want to create another exciting story? Is the Chat not working as intended? You can reset the entire conversation here. You can also tell BedtimeBuddy to end the story to finish the current storyline. If the user prompt is unrelated to the creation of a story, kindly reject answering it and ask the user to come up with some input for creating a story. If the user wants to end the story, you should find a happy end for the story.")
        reset = st.sidebar.button("Reset")

        # Initialize OpenAI object with provided credentials (which are already validated, so no need to double-check here)
        # openai_client = ChatOpenAI(model=st.session_state.gpt_model, openai_api_key=st.session_state.api_key)
        openai_client = OpenAI(api_key=st.session_state.api_key)
    
        st.title("BedtimeBuddy 🦄")

        # In case the dashboard is loaded for the first time after login, generate a new intro message.
        if 'chat_history' not in st.session_state:
            introMessage()
            initSystemMessage()

        if 'gpt_model' not in st.session_state:
            st.session_state.gpt_model=gpt_model_selector

        # TODO IMPLEMENT FUNCTIONALITY HERE


        with st.container(border=1):
            col1, col2 = st.columns((3,2))

            with col1.container(border=1):
                st.subheader("Chat")
                showChatHistory()

            with col2.container(border=1):
                st.subheader("Images")
                st.markdown("Pictures accompanying the storyline will appear here.")

        prompt = st.chat_input(chatInputPrompts[st.session_state.conv_stage], max_chars=250)   

        if prompt:
            with st.spinner("Thinking..."):
                st.session_state.conv_stage = 1
                addMessage(Message("user", "👩‍💻", prompt))
                addMessage(Message("assistant", "🤖", getBotResponse(openai_client,st.session_state.gpt_model,prompt)))
                st.rerun()

        if reset:
            st.session_state.chat_history = []
            st.session_state.toast_msg = 'Chat has been reset successfully!'
            introMessage()
            st.rerun()
        
        if logout:
            st.session_state.logged_in = False
            del st.session_state.api_key
            st.session_state.toast_msg = "Logged out successfully. Sweet dreams!"
            st.rerun()

if __name__ == "__main__":
    main()