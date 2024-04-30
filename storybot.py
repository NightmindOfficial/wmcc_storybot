# Note: I am using the plugin 'Better Comments' for easier comment legibility. This is why some comments might have additional asterisks, exclamation marks, etc.
# I can highly recommend the plugin if you are working in VS Code (chances are high you are already using it..)

### * 02 IMPORTS * ###
# All these imports are needed to either display the app correctly, make requests to the API, or to work with OpenAI.

import random
import numpy as np
import streamlit as st # Streamlit App functionality
import requests # API Requests
from time import sleep
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains import LLMChain
from IPython.display import Image, display


### * 01 FINAL VARIABLES * ###
# This section contains all final variables referenced at a later stage in the program.

setup_link = "https://platform.openai.com/docs/quickstart/account-setup"

# To be able to initialize the session state variables in a loop, this list references all keys to be initialized.
sessionStateKeys = ['logged_in', 'toast_msg', 'conversation_stage', 'dalle_task', 'prompt_disabled']

# The app allows the user to choose the GPT model it should use for generating the stories. The options, however, are limited to these three models.
gpt_model_options = ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',]
dalle_model_options = ['dall-e-2', 'dall-e-3']

# TODO add explainer
chatInputPrompts = [
    'Start off your journey by entering the first few lines of a story, or give me a general theme.',
    'How should the story continue?',
    'How should the story continue? Use one of the suggestions above, or write some manual input.'
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


### * PROMPT TEMPLATES * ###
# These templates are used by langchain to invoke prompts on the selected LLM. As the different stages of the chat have different requirements, I use different prompt templates.
# The prompt templates are returned by a function called 'getPromptTemplate' based on the conversation stage the chat is in.

promptTemplates = [
    PromptTemplate( # * STAGE 0 - START OF A STORY, ASKING FOR FOLLOW-UP (NO BUTTONS)
        input_variables=["user-prompt","chat-history"],
        template="""
        You are a chatbot whose sole purpose is to write bedtime stories for younger children. If the user input is not related to a story, you kindly direct them to giving input for a bedtime story.
        Your output must at all times be child-friendly, easy to understand for a young audience, and exciting to read. 
        It should not contain difficult words and should be written in a bedtime story style.
        Your answers should be structured in a JSON format, and include the following keys: 'story', 'dalle-prompt'.
        'story' is your story fragment, which must contain a small cliffhanger at the end to allow the story to be continued. Only include the BEGINNING OF THE STORY in the value for 'story' together with a friendly question to the user asking for input on how the story should continue. 
        'dalle-prompt' is a shorter prompt summarizing the current story part for a future dall-e prompt.

        User Input: {user-prompt}
        """,
    ),
    PromptTemplate( # * STAGE 1 -  CONTINUATION OF A STORY, ASKING FOR FOLLOW-UP (WITHOUT BUTTONS)
        input_variables=["user-prompt","chat-history"],
        template="""
        You are a chatbot whose sole purpose is to write bedtime stories for younger children. If the user input is not related to a story, you kindly direct them to giving input for a bedtime story.
        Your output must at all times be child-friendly, easy to understand for a young audience, and exciting to read. 
        It should not contain difficult words and should be written in a bedtime story style.
        Your answers should be structured in a JSON format, and include the following keys: 'story', 'dalle-prompt'.
        'story' is your story fragment, which must contain a small cliffhanger at the end to allow the story to be continued. The story MUST relate to the story parts previously generated during the conversation listed under "chat history" and must not contain any plotholes.
        'dalle-prompt' is a shorter prompt summarizing the current story part for a future dall-e prompt.

        User Input: {user-prompt}

        Chat History: {chat-history}
        """,
    ),
    PromptTemplate( # * STAGE 2 - CONTINUATION OF A STORY, ASKING FOR FOLLOW-UP (WITH BUTTONS)
        input_variables=["user-prompt","chat-history"],
        template="""
        You are a chatbot whose sole purpose is to write bedtime stories for younger children. If the user input is not related to a story, you kindly direct them to giving input for a bedtime story.
        Your output must at all times be child-friendly, easy to understand for a young audience, and exciting to read. 
        It should not contain difficult words and should be written in a bedtime story style.
        Your answers should be structured in a JSON format, and include the following keys: 'story', 'dalle-prompt', 'opt1', 'opt2', 'opt3'.
        'story' is your story fragment, which must contain a small cliffhanger at the end to allow the story to be continued. The story MUST relate to the story parts previously generated during the conversation listed under "chat history" and must not contain any plotholes.
        'dalle-prompt' is a shorter prompt summarizing the current story part for a future dall-e prompt. 
        The 'opt1', 'opt2', 'opt3' keys should contain keywords of 1 to 3 words, suggesting how the story could continue.
        IF YOU FIND THAT THE USER MENTIONS TO END THE STORY IN THE CHAT HISTORY, WRITE A HAPPY END WITH NO FURTHER CLIFFHANGERS INSTEAD FOR THE 'STORY' VALUE AND MAKE SURE TO END ON "THE END".

        User Input: {user-prompt}

        Chat History: {chat-history}
        """,
    ),
]


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
def getPromptTemplate(conversation_stage):
    return promptTemplates[conversation_stage]


# TODO Add Explainer Comment
def showChatHistory():
    if 'chat_history' in st.session_state:
        for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

# TODO Add Explainer Comment
def addMessage(role, content):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': role, 'content': content})

# TODO Add Explainer Comment
def getBotResponse(client,user_prompt):
        prompt = getPromptTemplate(st.session_state.conv_stage)
        chat_history = st.session_state.chat_history[1:] # Do not include the first (hard-coded) intro message
        chain = prompt | client | JsonOutputParser()
        # NOTE: I intentionally decided not to use a stream here. It does not make too much sense streaming an (incomplete) JSON object. Since the story pieces are not
        # too long and the user is notified of the "thinking" with a spinner, this does not impair the user experience too much in my opinion.
        output = chain.invoke({
            "chat-history": chat_history,
            "user-prompt": user_prompt
        })
        return output


# TODO Add Explainer Comment
def showImages():
    if 'image_urls' in st.session_state:
        for image in st.session_state.image_urls:
                st.image(image["url"], caption=image["caption"], use_column_width=True)


# TODO Add Explainer Comment
def addImage(url, caption):
    if 'image_urls' not in st.session_state:
        st.session_state.image_urls = []
    st.session_state.image_urls.append({'url': url, 'caption': caption})


# TODO Add Explainer Comment
def generateImage(client, prompt):
    response = client.images.generate(
        model=st.session_state.dalle_model,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        )
    
    return response.data[0].url

def submitPrompt(prompt, chat_openai_client):
    addMessage("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Writing in progress, hold tight..."):
            try:
                response = getBotResponse(chat_openai_client,prompt)
                addMessage("assistant", response["story"])

                st.session_state.dalle_task = response["dalle-prompt"]
                st.session_state.conv_stage = min(2, st.session_state.conv_stage + 1)
                st.session_state.prompt_disabled = True
                # ! remove before flight.
                print(response)

                if (st.session_state.conv_stage == 2):
                    for i in range(1, 4):
                        st.session_state.prompt_buttons.append(response[f"opt{i}"])
                st.rerun()
            except Exception as e:
                st.error(f"Whoops, something did not work out as expected. Maybe your input violated a content policy? You can try again and see if the next attempt runs smoothly. {e}")


# Returns a hard-coded greeting message for the user. Using the OpenAI interface for this would be inefficient, since the text is not helpful for generating the messages.
def introMessage():
    addMessage("assistant", aiIntroMessages[random.randint(0,len(aiIntroMessages)-1)])



### * 04 MAIN FUNCTION * ###
# Where all the frontend layouting and function calling happens. This is the core of my streamlit application.

def main():

    # Set the streamlit layout to 'wide' to get more space, and make sure the sidebar is visible per default.
    st.set_page_config(page_title="BedtimeBuddy", layout="wide", initial_sidebar_state="expanded")

    # Streamlit Sidebar - if the user is not logged in, not much will be displayed here. As soon as the user is logged in, they will have the option to log out again and tweak some settings.
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

                # * Sets the conversation stage to 0. This means that the chat has just started, affecting which prompt template will be used as well as the wording of the chat input placeholder.
                # * Possible values: 0 (initial stage - first iteration without buttons), 1 (continuation - without button as options), 2 (continuation - WITH buttons)
                st.session_state.conv_stage = 0
                st.rerun()
            else:
                st.error(f"Whoops, that didn't work. Have you checked if the key is correct? (Error: {message})")

    # * LOGGED-IN DASHBOARD / MAIN SCREEN if the user has logged in
    else:

        st.sidebar.success("You are logged in.")
        st.sidebar.write("Feeling sleepy? Log out to prevent unauthorized use. Your API key will not be stored permanently in this program.")
        logout = st.sidebar.button("Log out")
        st.sidebar.divider()

        st.sidebar.subheader("Reset Conversation")
        st.sidebar.write("Want to create another exciting story? Is the Chat not working as intended? You can reset the entire conversation here. You can also tell BedtimeBuddy to end the story to finish the current storyline.")
        reset = st.sidebar.button("Reset")
        st.sidebar.divider()

        st.sidebar.subheader("GPT Model Selector")
        st.sidebar.write("You can choose the GPT model used for generating the bedtime stories and images.")
        gpt_model_selector = st.sidebar.selectbox(
            'GPT-model in use:',
            gpt_model_options,
            index=0, # gpt-4-turbo is the default model
            help="GPT-4 models typically yield better results, but at a higher cost.",
        )
        dalle_model_selector = st.sidebar.selectbox(
            'DALL-E-model in use:',
            dalle_model_options,
            index=1, # dall-e-3 is the default model
            help="Dall-E 2 costs approx. 0.02\$/image, DALL-E 3 about 0.04\$/image.",
        )
        st.sidebar.divider()

        st.sidebar.caption(f"Debug: Stage {st.session_state.conv_stage}")

        # Initialize the model choice in the st session_state if not done already
        if 'gpt_model' not in st.session_state:
            st.session_state.gpt_model=gpt_model_selector

        if 'dalle_model' not in st.session_state:
            st.session_state.dalle_model=dalle_model_selector

        # In case the dashboard is loaded for the first time after login, generate a new intro message.
        if 'chat_history' not in st.session_state:
            introMessage()

        if 'image_urls' not in st.session_state:
            st.session_state.image_urls = []

        if 'prompt_buttons' not in st.session_state:
            st.session_state.prompt_buttons = []


        # Initialize OpenAI object with provided credentials (which are already validated, so no need to double-check here)
        dalle = OpenAI(api_key=st.session_state.api_key)
        chat_openai_client = ChatOpenAI(model=st.session_state.gpt_model, openai_api_key=st.session_state.api_key)

    
        st.title("BedtimeBuddy 🦄")

        col1, col2 = st.columns((75,25))

        with col1.container(border=1):
            st.subheader("Chat")
            showChatHistory()

            if (st.session_state.conv_stage == 2 and st.session_state.prompt_buttons != []):
                st.markdown("**Here's what could happen:**")
                btncol1, btncol2, btncol3 = st.columns(3)
                with btncol1:
                    st.button(st.session_state.prompt_buttons[0], disabled=st.session_state.prompt_disabled, on_click=submitPrompt, args=[st.session_state.prompt_buttons[0], chat_openai_client])
                with btncol2:
                    st.button(st.session_state.prompt_buttons[1], disabled=st.session_state.prompt_disabled, on_click=submitPrompt, args=[st.session_state.prompt_buttons[1], chat_openai_client])
                with btncol3:
                    st.button(st.session_state.prompt_buttons[2], disabled=st.session_state.prompt_disabled, on_click=submitPrompt, args=[st.session_state.prompt_buttons[2], chat_openai_client])

            prompt = st.chat_input(chatInputPrompts[st.session_state.conv_stage], max_chars=250, disabled=st.session_state.prompt_disabled)
            

            if (prompt != None) and (prompt.strip() != ""):
                submitPrompt(prompt, chat_openai_client)

        with col2.container(border=1):
            st.subheader("Images")
            st.markdown("Pictures accompanying the storyline will appear here.")
            showImages()

            # Checks every rerun if a toast message has been stored prior to executing the rerun, and shows the toast if necessary.
            if st.session_state.dalle_task != False:
                with st.spinner("Creating a stunning Picture..."):
                    try:
                        addImage(generateImage(dalle, st.session_state.dalle_task), st.session_state.dalle_task)
                        st.session_state.prompt_disabled = False
                        st.session_state.dalle_task = False
                        st.rerun()
                    except Exception as e:
                            st.error(f"Whoops, something did not work out as expected. Maybe your input violated a content policy? You can try again and see if the next attempt runs smoothly. {e}")


        st.info("Attention Parents: BedtimeBuddy is made for kids. Our bot will not consider input that is not appropriate for children or that is against OpenAI's content policies. Nevertheless, the bot can make mistakes, and any resemblance to real events or people is coincidental. Parental guidance is advised.")


        if reset:
            st.session_state.chat_history = []
            st.session_state.image_urls = []
            st.session_state.prompt_buttons = []
            st.session_state.dalle_task = False
            st.session_state.conv_stage = 0
            st.session_state.toast_msg = 'Chat has been reset successfully!'
            introMessage()
            st.rerun()
        
        if logout:
            st.session_state.logged_in = False
            del st.session_state.api_key
            st.session_state.toast_msg = "Logged out successfully. Sweet dreams!"
            st.rerun()


# Lastly, call the main function of the script.
if __name__ == "__main__":
    main()