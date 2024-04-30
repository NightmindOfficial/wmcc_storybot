#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# File: WMCC Storybot
# Author: Otis Mohr

# * Note: I am using the plugin 'Better Comments' for easier comment legibility. This is why some comments might have additional asterisks, exclamation marks, etc.
# * I can highly recommend the plugin if you are working in VS Code (chances are high you are already using it..)

### * 01 IMPORTS * ###
# All these imports are needed to either display the app correctly, make requests to the API, work with OpenAI, generate custom prompts, 

import random # Random Number generator for choosing a random intro message
import streamlit as st # Streamlit App functionality
import requests # API Requests
from openai import OpenAI # OpenAI API Interface
from langchain_openai import ChatOpenAI # OpenAI for Langchain Functions
from langchain.prompts import PromptTemplate # Prompt Template Package for custom Prompting
from langchain_core.output_parsers import JsonOutputParser # Package for interpreting OpenAI responses as JSON objects for further processing
import tiktoken # Helper package to calculate # of tokens needed


### * 02 FINAL VARIABLES * ###
# This section contains all final variables referenced at a later stage in the program.

setup_link = "https://platform.openai.com/docs/quickstart/account-setup" # Used in the login page to guide the user to create an API key, if not already done so

# To be able to initialize the session state variables in a loop, this list references all keys to be initialized.
sessionStateKeys = ['logged_in', 'toast_msg', 'conversation_stage', 'dalle_task', 'prompt_disabled', 'prompt_callback']

# The app allows the user to choose the GPT model it should use for generating the stories. The options, however, are limited to these three (two) models to keep things relatively simple.
gpt_model_options = ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
dalle_model_options = ['dall-e-2', 'dall-e-3']

# These are the placeholder prompts used in the chat input field. Depending on the conversation stage, the messages ask the user to do different things.
chatInputPrompts = [
    'Start off your journey by entering the first few lines of a story, or give me a general theme.',
    'How should the story continue?',
    'How should we continue? Use one of the suggestions above, or write some manual input.'
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


### * 03 PROMPT TEMPLATES * ###
# These templates are used by langchain to invoke prompts on the selected LLM. As the different stages of the chat have different requirements, I use different prompt templates.
# The prompt templates are returned by a function called 'getPromptTemplate' based on the conversation stage the chat is in.

promptTemplates = [
    PromptTemplate( # STAGE 0 - START OF A STORY, ASKING FOR FOLLOW-UP (NO BUTTONS)
        input_variables=["userprompt","chathistory"],
        template="""
        You are a chatbot whose sole purpose is to write bedtime stories for younger children. If the user input is not related to a story, you kindly direct them to giving input for a bedtime story.
        Your output must at all times be child-friendly, easy to understand for a young audience, and exciting to read. 
        It should not contain difficult words and should be written in a bedtime story style.
        Your answers should be structured in a JSON format, and include the following keys: 'story', 'dalle-prompt'.
        'story' is your story fragment, which must contain a small cliffhanger at the end to allow the story to be continued. Only include the BEGINNING OF THE STORY in the value for 'story' together with a friendly question to the user asking for input on how the story should continue. 
        'dalle-prompt' is a shorter prompt summarizing the current story part for a future dall-e prompt.

        User Input: {userprompt}
        """,
    ),
    PromptTemplate( # STAGE 1 - CONTINUATION OF A STORY, ASKING FOR FOLLOW-UP (WITH BUTTONS)
        input_variables=["userprompt","chathistory"],
        template="""
        You are a chatbot whose sole purpose is to write bedtime stories for younger children. If the user input is not related to a story, you kindly direct them to giving input for a bedtime story.
        Your output must at all times be child-friendly, easy to understand for a young audience, and exciting to read. 
        It should not contain difficult words and should be written in a bedtime story style.
        Your answers should be structured in a JSON format, and include the following keys: 'story', 'dalle-prompt', 'opt1', 'opt2', 'opt3'.
        'story' is your story fragment, which must contain a small cliffhanger at the end to allow the story to be continued. The story MUST relate to the story parts previously generated during the conversation listed under "chat history" and must not contain any plotholes.
        'dalle-prompt' is a shorter prompt summarizing the current story part for a future dall-e prompt. 
        The 'opt1', 'opt2', 'opt3' keys should contain keywords of 1 to 3 words, suggesting how the story could continue.
        IF YOU FIND THAT THE USER MENTIONS TO END THE STORY IN THE CHAT HISTORY, WRITE A HAPPY END WITH NO FURTHER CLIFFHANGERS INSTEAD FOR THE 'STORY' VALUE AND MAKE SURE TO END ON "THE END". IN THIS CASE, DO NOT INCLUDE ANY 'OPT' KEYS IN YOUR ANSWER.

        User Input: {userprompt}

        Chat History: {chathistory}
        """,
    ),
]


### * 04 FUNCTIONS * ###
# Helper functions and handling of chat requests, summarizing the contents, generating images, etc.

# Function to check the validity of the OpenAI API key, rejecting the key if it is not valid in the first place (thus restricting user access to the main part of the program)
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


# Helper Function that takes text input from the session_state and uses it to display a toast widget in streamlit. 
# Unless otherwise specified, a checkbox icon will be provided automatically.
# The function is build this way because toast messages would be cancelled by app reruns, 
# thus, I added a 'listener' in the main function that triggers this function if the respective session_state variable is defined, and self-resets the variable to avoid infinite loops.
def display_toast_msg(icon='✅'):
    st.toast(st.session_state.toast_msg, icon=icon)
    st.session_state.toast_msg = False
    return

# Helper function that returns the correct prompt template based on the conversation stage (0 or 1)
def getPromptTemplate(conversation_stage):
    return promptTemplates[conversation_stage]

# Helper function that returns the correct chat input placeholder text based on the conversation stage (depending on what the user should do, they get different call-to-actions)
def getChatInputPrompt():
    if (st.session_state.conv_stage == 0):
        return chatInputPrompts[0] # 'Start off your journey by entering the first few lines of a story, or give me a general theme.'
    elif(st.session_state.conv_stage == 1 and not st.session_state.prompt_buttons == []): # If the user can choose between predefined follow-up prompts, the placeholder takes that into account, too
        return chatInputPrompts[2] # 'How should we continue? Use one of the suggestions above, or write some manual input.' 
    return chatInputPrompts[1] # 'How should the story continue?'


# Function that looks for previous interactions between user and assistant, and creates a pre-styled chat message object for each interaction.
def showChatHistory():
    if 'chat_history' in st.session_state:
        for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

# Function that consults OpenAI's moderation interface to check the user input for possible policy violations. If flagged for any violations, the function will return True.
def checkContentViolation(client, prompt):
    response = client.moderations.create(input=prompt)
    return response.results[0].flagged

# Function that adds the latest user prompt to the chat history, persisting it in the session_state. In the runtime of the program, function calls for adding user messages are
# always preceded by checkContentViolation(). Content violations concerning the assistant output are caught elsewhere.
def addMessage(role, content):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': role, 'content': content})

# Function that consults the completion interface of the selected OpenAI model. This is done by invoking a custom langchain, which modifies the original prompt with a
# PromptTemplate and uses a JSONOutputParser() to interpret the JSON-styled output of the LLM, returning the parsed output.
def getBotResponse(client,user_prompt):
        prompt = getPromptTemplate(st.session_state.conv_stage)
        chat_history = reduceChatHistoryLength(user_prompt)
        
        chain = prompt | client | JsonOutputParser()
        # NOTE: I intentionally decided not to use a stream here. It does not make too much sense streaming an (incomplete) JSON object. Since the story pieces are not
        # too long and the user is notified of the "thinking" with a spinner, this does not impair the user experience too much in my opinion.
        output = chain.invoke({
            "chathistory": chat_history,
            "userprompt": user_prompt
        })
        return output


# Similar to showChatHistory(), this is a function that looks for previously generated story images, and creates a streamlit image widget for each image url persisted in the session_state.
def showImages():
    if 'image_urls' in st.session_state:
        for image in st.session_state.image_urls:
                st.image(image["url"], caption=image["caption"], use_column_width=True)


# Similar to addMessage(), this is a function that adds the latest image url to the image_urls variable, persisting it in the session_state.
def addImage(url, caption):
    if 'image_urls' not in st.session_state:
        st.session_state.image_urls = []
    st.session_state.image_urls.append({'url': url, 'caption': caption})


# Function that consults the OpenAI images interface of the selected OpenAI (DALL-E) model. The image size is fixed at 1024x1024, being the smallest common image size for DALL-E Versions 2 and 3.
# Note that only the image url is returned, not the actual image object.
def generateImage(client, prompt):
    response = client.images.generate(
        model=st.session_state.dalle_model,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        )
    
    return response.data[0].url


# Helper Function that cicrumvents a limitation of Streamlit not allowing reruns in a button callback (since a callback is called before a rerun already).
# It saves the user-selected keyword prompt (from pressing one of the three buttons offered) in a session_state variable and resets the button list.
# After the callback, a listening loop in the main function will pick up that prompt_callback is defined, and trigger the getBotResponse() function with the saved prompt.
# There are certainly cleaner ways to solve this, but all previous attempts to generate the response within the callback resulted in the app layout being destroyed.
def buttonCallback(prompt):
    st.session_state.prompt_buttons = []
    st.session_state.prompt_callback = prompt
    # No rerun here, since this is a callback function.

# Function that returns a hard-coded greeting message for the user upon starting a new chat. 
# Using the OpenAI interface for this would be inefficient, since the text is not helpful for generating the messages, and thus should not be part of the prompt.
# The function is called each time the chat is reset, always being the first message in the chat_history in the session_state. As mentioned, it will NOT be considered for the prompt (decorative message)
# The function will return one of the pre-defined greetings randomly, to avoid making the greeting too boring for the user after several iterations, thereby somewhat imitating a generated response.
def introMessage():
    addMessage("assistant", aiIntroMessages[random.randint(0,len(aiIntroMessages)-1)])


# Helper Function to count the number of tokens in a given text with the tiktoken module.
def count_tokens(input):
    encoding = tiktoken.encoding_for_model(st.session_state.gpt_model)
    encodedString=encoding.encode(input)
    return len(encodedString)

# Function to manage the chat history length considering the token length of the total prompt, keeping within specified token limits.
# The function is called by getBotResponse(), returning an adapted version of the chat history that together with the selected custom prompt will never exceed a pre-determined amount of tokens.
def reduceChatHistoryLength(user_prompt, max_tokens=4000):
    prompt = getPromptTemplate(st.session_state.conv_stage)
    chat_history = st.session_state.chat_history[1:] # Do not include the first (hard-coded) intro message


    total_text = " ".join(prompt.format(chathistory = chat_history, userprompt= user_prompt))
    total_tokens = count_tokens(total_text)

    # Truncate chat history if the total tokens exceed the maximum allowed
    while total_tokens > max_tokens and len(chat_history) > 1:
        # Remove oldest entries until under limit
        removed_text = chat_history.pop(0)
        total_text = " ".join(prompt.format(chathistory = chat_history, userprompt= user_prompt))
        total_tokens = count_tokens(total_text)
    st.session_state.prompt_token_len = total_tokens
    return chat_history




### * 04 MAIN FUNCTION * ###
# Where all the frontend layouting and function calling happens. This is the core of my streamlit application.

def main():

    # Set the streamlit layout to 'wide' to get more space, and make sure the sidebar is visible per default.
    st.set_page_config(page_title="BedtimeBuddy", layout="wide", initial_sidebar_state="expanded")

    # Streamlit Sidebar - if the user is not logged in, not much will be displayed here. As soon as the user is logged in, they will have the option to log out again and tweak some settings.
    st.sidebar.title("Login Status")

    # Initialize Session state variables at the first run of the script (when they have not been initialized yet). Setting them to False works for most keys in the beginning, as they will soon take on other values.
    for key in sessionStateKeys:
        if key not in st.session_state:
            st.session_state[key] = False

    # Listening Loop - Checks every rerun if a toast message has been stored prior to executing a rerun, and shows the toast if one is defined.
    # Since display_toast_msg() resets the toast message in the session_state, there is no danger of running into an infinite loop.
    if st.session_state.toast_msg != False:
        display_toast_msg()

    # Initializing the prompt token length in session_state. This is just for debug purposes, and to make it easier to check if the max number of tokens are adhered to.
    if 'prompt_token_len' not in st.session_state:
        st.session_state.prompt_token_len = 0


    ### * STATE 1 --- LOGIN / WELCOME PAGE if the user is not logged in / has just logged out * ###
    # If the user is not logged in yet (has not provided their API key yet), the app will ask the user to provide one in order to use the app.

    if not st.session_state.logged_in:

        st.sidebar.info("You are not logged in. Please provide a valid OpenAI API key to continue.")
        
        st.title("Welcome to BedtimeBuddy 🦄")
        st.subheader("Your Nightly Dose of ✨ Dream Dust! ✨")
        st.write("Attention Parents: To start diving into your next adventure bedtime story with your kids, please provide a valid API key for OpenAI.")
        st.info("First time loggin in, and not sure how to get an API key? [Check out the OpenAI Documentation!](%s)" % setup_link ) #Direct Link to the OpenAI API help page

        # Ask for the API key of the user, which will then temporarily be stored in the session state for the session's duration.
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")

        # Button that handles the login request of the user, using the check_api_key function to check the validity of the key,
        # before moving on to the logged-in state if the key is actually valid.

        # Handle Login - this loop will theck the validity of the API key provided and persist it in the session_state if it is valid.
        if st.button("Login"):
            isValid, message = check_api_key(api_key)
            if isValid:
                st.session_state.logged_in = True # This also changes the app layout from 'logged out' to 'logged in'
                st.session_state.api_key = api_key # API Key is now accessible in the session_state...
                del api_key # and is not needed in the controller variable anymore
                st.session_state.toast_msg = "Logged in successfully. Let's create a great story together!" # Toast to be displayed on the next rerun

                # * IMPORTANT! The following code sets the 'conversation stage' to 0. This means that the chat has just started, affecting which prompt template will be used as well as the wording of the chat input placeholder.
                # * Possible values: 0 (initial stage - first iteration without buttons), 1 (continuation - with buttons)
                st.session_state.conv_stage = 0
                st.rerun() # Reload the page for the app layout changes to take effect
            else:
                st.error(f"Whoops, that didn't work. Have you checked if the key is correct? (Error: {message})") # In any other case, show an error message telling the user why the login failed (e.g., in case the API key was invalid)



    ### * STATE 2 --- LOGGED-IN DASHBOARD / MAIN SCREEN if the user has logged in * ###
    # If the user is logged in, the "actual" app layout will become accessible. Here the user can chat with our custom chatbot in the chat column, and look at generated pictures
    # in the pictures column. On the sidebar, various settings and controls will become accessible.
    else:

        # ** SIDEBAR ** #
        st.sidebar.success("You are logged in.")
        st.sidebar.write("Feeling sleepy? Log out to prevent unauthorized use. Your API key will not be stored permanently in this program.")
        logout = st.sidebar.button("Log out") # Function handling the logout process can be found further below
        st.sidebar.divider()

        st.sidebar.subheader("Reset Conversation")
        st.sidebar.write("Want to create another exciting story? Is the Chat not working as intended? You can reset the entire conversation here. You can also tell BedtimeBuddy to end the story to finish the current storyline.")
        reset = st.sidebar.button("Reset") # Function handling the chat reset process can be found further below
        st.sidebar.divider()

        # To allow further customization and save money during testing, I implemented an extra functionality allowing the user to choose the GPT models used in creating
        # the response texts and images. These can be customized here.
        st.sidebar.subheader("GPT Model Selector")
        st.sidebar.write("You can choose the GPT model used for generating the bedtime stories and images.")
        gpt_model_selector = st.sidebar.selectbox(
            'GPT-model in use:',
            gpt_model_options, # Set of options pre-defined above. Options are: ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
            index=0, # gpt-4-turbo is the default model
            help="GPT-4 models typically yield better results, but at a higher cost.",
        )
        dalle_model_selector = st.sidebar.selectbox(
            'DALL-E-model in use:',
            dalle_model_options, # Set of options pre-defined above. Options are: ['dall-e-2', 'dall-e-3']
            index=1, # dall-e-3 is the default model
            help="Dall-E 2 costs approx. 0.02\$/image, DALL-E 3 about 0.04\$/image.",
        )
        st.sidebar.divider()

        st.sidebar.caption(f"Debug: Stage {st.session_state.conv_stage}, Token Length {st.session_state.prompt_token_len}") # Initially for debug purposes, decided to keep it since it might be interesting


        # ** INITIALIZATIONS **
        # Initialize the model choices in the st session_state regularly on app rerun
        st.session_state.gpt_model=gpt_model_selector
        st.session_state.dalle_model=dalle_model_selector

        # In case the dashboard is loaded for the first time after login, generate a new intro message and persist it in the session_state (will be handled by the introMessage() function)
        if 'chat_history' not in st.session_state:
            introMessage()

        # In case the dashboard is loaded for the first time after login, create an empty list for images to be generated
        if 'image_urls' not in st.session_state:
            st.session_state.image_urls = []

        # In case the dashboard is loaded for the first time after login, create an empty list for the keyword prompt buttons to be generated
        if 'prompt_buttons' not in st.session_state:
            st.session_state.prompt_buttons = []

        # Initialize OpenAI objects with provided credentials (which are already validated, so no need to double-check here)
        dalle = OpenAI(api_key=st.session_state.api_key) # For image generation and content moderation (validation)
        chat_openai_client = ChatOpenAI(model=st.session_state.gpt_model, openai_api_key=st.session_state.api_key) # For response generation via langchain

        # * MAIN APP LAYOUT * #
        st.title("BedtimeBuddy 🦄")

        col1, col2 = st.columns((75,25)) # Chat window is the primary focus, images are generated on the side

        # * CHAT WINDOW * #
        with col1.container(border=1):
            st.subheader("Chat")
            showChatHistory()

            # Function responsible for submitting a new prompt to the chat and for generating and displaying the corresponsing AI response.
            # Although most functions are declared in a different part of the program, I figured that placing this elsewhere would mess with the
            # creation of chat message previews - thus, unfortunately - this needs to stay here in order for the program to display messages correctly.
            def submitPrompt(prompt, chat_openai_client):
                try: # Since a lot can go wrong in creating a prompt (e.g., service not available, lack of funds, response policy violations...), I decided to catch these errors altogether.
                    if checkContentViolation(dalle,prompt): # If the user input vialates the moderation policy of OpenAI, the prompt will not be considered and a warning will be displayed.
                            st.warning("Sorry, this prompt violates OpenAI's content policies and might not be suitable for children. It has not been added to your chat history. Please choose a different prompt.")
                    else: # In all other cases, the message will be added to the chat_history, but since the rerun will happen later, a live 'preview' widget will be created, too.
                        addMessage("user", prompt)
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # The response creation is going to take some time. Since we cannot stream the response because we expect a JSON object, a spinner is displayed informing
                        # the user that the bot is working, thereby enhancing the user experience.
                        with st.chat_message("assistant"):
                            with st.spinner("Writing a great story, hold tight..."):

                                    # Then, the actual response generation mechanism is triggered.
                                    response = getBotResponse(chat_openai_client,prompt)
                                    # In case that is successful, we add the response to the chat_history.
                                    addMessage("assistant", response["story"])

                                    # Next up is the generation of an image accompanying the story fragment. But we do not want the user to wait any longer to read the text.
                                    # Thus, I am performing a similar trick as with the toast message listener: The "Task" of generating a picture is persisted in the session_state,
                                    # and an app rerun is triggered. As soon as the app reruns, a listener in the main function will notice that the dalle_task variable is defined,
                                    # and will trigger the generation of t a dall-e image. While this is happening, the story bit is already visible for the user to read, thus reducing
                                    # the waiting time by splitting it up into two tasks.
                                    st.session_state.dalle_task = response["dalle-prompt"]
                                    # I will also turn off the editing capabilities of the chat input to avoid further user inputs before the image is done generating.
                                    st.session_state.prompt_disabled = True

                                    # If the conversation stage is already at 1, this means that the GPT model has generated some keyword options on how the story should continue.
                                    # The following code will persist these options in the session_state so that they will be rendered as buttons above the chat input on the next app rerun.
                                    if (st.session_state.conv_stage == 1):
                                        st.session_state.prompt_buttons = []
                                        try:
                                            for i in range(1, 4):
                                                if response[f"opt{i}"] == '': # Sometimes, gpt-3 generates empty option values for the buttons when it is told to write a story end. I am counting this as a story end and will reset the conversation stage to 0, making way for the next story to be created.
                                                    raise Exception
                                                else:
                                                    st.session_state.prompt_buttons.append(response[f"opt{i}"]) # Persist the currently generated set of options for later rendering as buttons
                                        except Exception as e:
                                            st.session_state.toast_msg = "No buttons are displayed, as the story reached its end (or an error occurred)."
                                            st.session_state.conv_stage = 0
                                            st.rerun()

                                    # If everything completes without errors, we have done at least one iteration with the chatbot, so the next time, the response should include
                                    # keyword suggestions for future prompts. Thus, updating the conversation stage to 1.
                                    st.session_state.conv_stage = 1

                                    # Finally, rerun the app to reflect all changes.
                                    st.rerun()
                # In case anything did not work out as intended, show an error message to the user, allowing them to retry the prompt.
                except Exception as e:
                        st.error(f"Whoops, something did not work out as expected. Maybe your input violated a content policy? You can try again and see if the next attempt runs smoothly. {e}")

            # Listening Function: If a button callback has been persisted, that means that the user has pressed a suggestion button. The following code triggers a bot response
            # based on the prompt of the button pressed, then resets the variable in session_state to avoid infinite loops.
            if st.session_state.prompt_callback:
                prompt = st.session_state.prompt_callback
                st.session_state.prompt_callback = False
                submitPrompt(prompt, chat_openai_client)

            # If the conversation stage is at 1 (at least one manual user input has happened already), and the previous bot responses has generated suggestion prompts, show them as buttons.
            if (st.session_state.conv_stage == 1 and st.session_state.prompt_buttons != []):
                st.markdown("**Here's what could happen:**")
                btncol1, btncol2, btncol3 = st.columns(3)
                with btncol1:
                    st.button(st.session_state.prompt_buttons[0], 
                                disabled=st.session_state.prompt_disabled, # The buttons are disabled if the image for the previous story piece is still generating.
                                on_click=buttonCallback, # If the button is clicked, the callback function will persist its value which will then be picked up by the listening function above
                                args=[st.session_state.prompt_buttons[0]], # Argument for the callback function (simply the suggested prompt of the button)
                                use_container_width=True)
                with btncol2:
                    st.button(st.session_state.prompt_buttons[1], 
                                disabled=st.session_state.prompt_disabled, 
                                on_click=buttonCallback, 
                                args=[st.session_state.prompt_buttons[1]], 
                                use_container_width=True)
                with btncol3:
                    st.button(st.session_state.prompt_buttons[2],
                                disabled=st.session_state.prompt_disabled, 
                                on_click=buttonCallback, 
                                args=[st.session_state.prompt_buttons[2]],
                                use_container_width=True)

            
            # Show a chat input at the bottom of the chat window, allowing user input (as long as there is no picture generation underway). 280 characters (a Twitter message) should be enough input.
            prompt = st.chat_input(getChatInputPrompt(), max_chars=280, disabled=st.session_state.prompt_disabled)
            
            # Only allow the user to submit non-empty prompts. Use the GPT model selected by the user to trigger a bot response.
            if (prompt != None) and (prompt.strip() != ""):
                submitPrompt(prompt, chat_openai_client)

        # * PICTURE WINDOW * #
        with col2.container(border=1):
            st.subheader("Images")
            st.markdown("Pictures accompanying the storyline will appear here.")
            # Include all images generated so far
            showImages()

            # Checks every rerun if a picture generation task has been stored prior to executing the rerun, and triggers that generation if necessary.
            # It resets the variables and also enables the chat message input again, so no threat of infinite loops.
            if st.session_state.dalle_task != False:
                # Since the image generation might also take some time, show the user a spinner widget to indicate the bot is working, thereby enhancing user experience
                with st.spinner("Creating a stunning Picture..."):
                    try:
                        addImage(generateImage(dalle, st.session_state.dalle_task), st.session_state.dalle_task)
                        st.session_state.prompt_disabled = False
                        st.session_state.dalle_task = False
                        st.rerun()
                    except Exception as e: # If something goes wrong, catch any exception and inform the user there was a problem with generating the image
                            st.error(f"Whoops, something did not work out as expected. Maybe your input violated a content policy? You can try again and see if the next attempt runs smoothly. {e}")

        # Disclaimer text - although the prompt template and moderation function should take care of most non-complying in- and output, I don't want to risk it.
        st.info("Attention Parents: BedtimeBuddy is made for a kids audience. Our bot will not consider input that is not appropriate for children or that is against OpenAI's content policies. Nevertheless, the bot can make mistakes, and any resemblance to real events or people is coincidental. Parental guidance is strongly advised.")


        # RESET BUTTON HANDLER - resets most (not all) session_state variables related to the current chat, then triggers an accompanying toast message and reruns the app.
        if reset:
            st.session_state.chat_history = []
            st.session_state.image_urls = []
            st.session_state.prompt_buttons = []
            st.session_state.dalle_task = False
            st.session_state.conv_stage = 0
            st.session_state.toast_msg = 'Chat has been reset successfully!'
            introMessage()
            st.rerun()
        
        # RESET BUTTON HANDLER - resets the entire session_state, then triggers an accompanying toast message and reruns the app.
        if logout:
            st.session_state.logged_in = False
            for key in st.session_state.keys():
                del st.session_state[key]

            st.session_state.toast_msg = "Logged out successfully. Sweet dreams!"
            st.rerun()


# Lastly, call the main function of the script.
if __name__ == "__main__":
    main()