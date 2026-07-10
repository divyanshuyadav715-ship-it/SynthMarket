# Use the official lightweight Python image
FROM python:3.10-slim

# Hugging Face Spaces require running as a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy files and set ownership to 'user'
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make the startup script executable
RUN chmod +x start.sh

# Expose the port Hugging Face expects
EXPOSE 7860

# Command to run both FastAPI and Streamlit
CMD ["./start.sh"]
