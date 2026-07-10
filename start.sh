#!/bin/bash

# 1. Start the FastAPI backend server in the background (&)
# It will run internally on port 8000 inside the container
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds to let the API start up properly
sleep 3

# 3. Start the Streamlit frontend dashboard in the foreground
# Hugging Face Spaces requires the main app to run on port 7860
streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0
