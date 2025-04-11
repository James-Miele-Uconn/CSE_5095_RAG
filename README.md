# CSE_5095_RAG
Run with "python RAG.py" <br />

Jupyter Notebook file also provided, though no simple way to run it is provided. <br />

Arguments for main file: <br />
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