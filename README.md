# WMCC Storybot

## What is this project about?

This project is part of my final assignment for a course on Web Mining and Cognitive Computing at my university.

## Installation

### Prerequisites

To run this app, you need a current version of `Python`, preferably `3.10` or higher.
You will also need `streamlit` for running the app, which can be installed by the package manager of your liking, e.g. with `pip install streamlit`.
Finally, you will need the `openai` package, which can be installed via the same way: `pip install openai`.
Other packages needed by the script should come with the base installation of your `Python` distribution.

You will also need your own OpenAI API key, which can be obtained on your [OpenAI Account page](https://platform.openai.com/docs/quickstart/account-setup).

### Running the script

To run this script on your local machine, you will need to clone this repository. After that, you can navigate in the folder and execute the app with:
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