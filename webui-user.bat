@echo off
cd rag
start cmd /k python rag_server.py
cd ..
start cmd /k python webui.py