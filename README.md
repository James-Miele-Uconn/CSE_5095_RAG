# CSE 5095 RAG System

To run the web-based UI, you need to start the backend RAG server and the frontend UI server. <br />
Use "(0) rag_server.bat" and "(0) webui-user.bat" to start the backend and frontend servers, respectively <br />
Alternatively, you can use the command-line version by calling "python RAG.py" <br />

A Jupyter Notebook version of the command-line option is also provided. <br />

Arguments for command-line version file: <br />
-h, --help <br />
&emsp; Show help options. <br />
--embedding <br />
&emsp; The embedding model to use. <br />
&emsp; Default is mxbai-embed-large. <br />
--model <br />
&emsp; The chat model to use. <br />
&emsp; Default is deepseek-r1:32b <br />
--num_docs <br />
&emsp; How many context document chunks to use. <br />
&emsp; Default is 5 chunks. <br />
--refresh_db <br />
&emsp; Force the database to be loaded from context files. <br />