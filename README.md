# CSE 5095 RAG System

This project is an attempt to make a general use RAG system.
The project contains two parts: the RAG server and the webui server.
The RAG server is a Flask server designed as a backend for this project.
The webui server is a gradio server designed as a frontend for this project.
For ease of use on a single machine, run both servers by using webui-user.bat.
Otherwise, you can run rag/rag_server.py for the RAG server, and webui.py for the webui server. <br />

# Overview of Use

The webui is split into three parts: the left column containing general settings for RAG system use, the main tabbed interface, and a customization sidebar on the right. The left column has most of the settings a user may want to change quickly; the main tabbed interface contains the chatbot interface and some controls for the chat history, the ability to manage topics and context files on the RAG server, and the ability to manage api keys per topic on the RAG server. The right sidebar contains some basic customization options.

# Left Column Explanation

The left column has several options that a user may want to modify in quick succession. The first option is the topic, which in essence is a directory on the RAG server. The topic determines the location of the directories for the context files, database files, and the file for the api keys. This allows separating different concepts into different topics, giving the user control over what topics should have access to what data. The ability to add and delete topics can be found in the "Manage Context Files" tab in the main interface.

The second option is the embedding model to use for the Chroma database. There are three main types as of now: models retrieved through Ollama (thus requiring the server to be running), models cloned from Huggingface (which use the Huggingface functions but do not need API keys), and models that are found online and need api keys. Currently, these options reflect the models I had been using for this project, but at some point it will (hopefully) change to reflect the models a user has downloaded/chosen to use.

The third option is similar to the second, but it is instead the LLM that will be used to generate the responses. A setup similar to that of the second option is given, so explanation for this option will end here.

The fourth set of options include general control over the usage of the chatbot. The first option in this set allows using the chatbot to summarize the chat history, adding this summary to the main prompt to be answered. The second option is whether the RAG system should be ignored; using this option allows interacting with the base LLM as a chatbot, without using the RAG system to pull information from the context documents. The third option is the number of document chunks to grab from the database when responding to the user; this is only relevant for the RAG system. The fourth option allows using the LLM to create a summary of each chunk used in the answer, providing the summary of all such summaries as the context information. This option and the first option are togglable due to the increase in time required for a response if using said options.

The fifth set of options in the left column involves rebuilding the database even if it otherwise would not be rebuilt. If said option is checked, variables that only concern the building of the database are shown; namely, the size of the chunks to be used and the overlap for each chunk (to be used with RecursiveCharacterTextSplitter) are shown.

# Main Interface Explanation

The main interface is comprised of three tabs: the RAG Chat tab, the Manage Context Files tab, and the Manage API Keys tab. The first of these tabs is the primary focus of the project. This tab allows interacting with the LLM, and also contains some basic controls for managing chat history. The current history can be saved (using a default name or the custom specified name) as a txt file. A txt file containing a python list of openai-style chat history messages can be uploaded to the chatbot, in which case the uploaded history will replace the current chat history.

In the Manage Context Files tab, the user can manage topics and the context files held within said topics. To add a topic, simply type a (Windows 10 friendly) directory name and press "Make New Topic"; the current topic will switch to the newly created topic. To delete the current topic (other than the default topic), press the "Delete Current Topic" button. To upload files to the RAG server for the current topic, add any desired files to the left "Upload Context Files" area (only txt, csv, and pdf are currently supported). The right "Context Files on Server" area shows which files have been uploaded to the RAG server for this topic; basic management of said files is provided.

In the Manage API Keys tab, a simple table is provided to allow using api keys for online models. Currently, only openai is supported, though there are plans to add in huggingface support as well. Additionally, to add an openai key, the "API" column must have "openai"; similarly, a huggingface key must have "huggingface". Ideally, this will be changed in the future.

# Customization Sidebar Explanation

The customization options are currently held in a collapsable sidebar; click the arrow on the top right of the screen to show/hide these options. The first two sets of options allow control over the icons used for the main chatbot interface. There is a default icon provided, more icons can be uploaded, all non-default icons can be deleted, and an icon can be chosen from the icons currently in the directory for the respective icon sets. A preview of the image to be used is shown.

There are two ways that the chatbot interface can present the messages: panel-style or bubble-style. There is a simple option to allow switching between these two styles as desired.

Similarly, there is support for both light-mode and dark-mode. The default mode will be based on your browser's preference, but a button is provided to allow switching between these two modes.

There are some options that require the webui server to restart in order to take effect. These options are currently the color to use as well as some basic controls over how the chatbot interface icons will be shown. To change one of these options, modify the desired option using the dropdown menu, press the "Reload UI" button, and refresh the tab.


<br><br><sup><sub>Favicon, chatbot default icon, user default icon designed by freepik</sub></sup>