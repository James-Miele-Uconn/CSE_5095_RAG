# CSE 5095 RAG System

To run the web-based UI, use webui-user.bat. <br />
Alternatively, for the command-line version use cmd-user.bat. <br />

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