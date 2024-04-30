# WMCC Storybot

## What is this project about?

This project is part of my final assignment for a course on Web Mining and Cognitive Computing at my university. See below for the original wording of the assignment. 

## Installation

### Prerequisites

To run this app, you need a current version of `Python`, preferably `3.10` or higher.

Furthermore, you will need to install the following packages:

| Package Name | Installation Command |
|:------------:|:--------------------:|
|´streamlit´ |´pip install streamlit´|
|´openai´|´pip install openai´|
|´langchain´|´pip install --upgrade langchain´|
|´langchain-openai´|´pip install --upgrade langchain-openai´|

You will also need `streamlit` for running the app, which can be installed by the package manager of your liking, e.g. with `pip install streamlit`.
Finally, you will need the `openai` package, which can be installed via the same way: `pip install openai`.
Other packages needed by the script should come with the base installation of your `Python` distribution.

You will also need your own OpenAI API key, which can be obtained on your [OpenAI Account page](https://platform.openai.com/docs/quickstart/account-setup).

### Running the script

To run this script on your local machine, you will need to clone this repository:
```
git clone https://github.com/NightmindOfficial/wmcc_storybot.git
```

After that, you can navigate in the folder and execute the app with:
```
streamlit run storybot.py
```

If everything is in order, your default browser should open the app automatically. In case you need to do this manually, use the following URL:
```
https://localhost:8501
```

**Note: The program will ask you to provide your own API key since I cannot reliably ensure everyone has an API key in their envs.**

## Feedback / Questions

You can contact me directly via DM if you encounter any issues or should you have any comments.

## Original Assignment

Create a streamlit app that automatically creates endless bedtime stories for kids. The app should be visually appealing and create textual and visual content.

The use case is as follows:

- Parent enters a text: Explain the scenery of a fairytale castle on a dark and stormy night.
- The LLM should provide the textual description and an image. You should visualize it properly in the app. Make it user-friendly.
- After text and image have been visualized, ask again for a follow-up:
- Parent enters: A knight is riding slowly towards the castle; explain this scenery and provide feasible options for how the story should continue.
- Now, the LLM should take into consideration what happened before, create a scenery text and image, and options for how to continue. The options should be made visible as a Button using just 1-3 keywords describing the options.
- The parent selects options and enters again the next step of the story.
- This is an iterative process until the parent/user tells the system to complete the story. The context windows (token length) should never exceed 4000 tokens. This means that you have to summarize the text in between using appropriate tools.
