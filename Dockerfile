# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# For HTTPS -- REMOVE IF DOES NOT WORK
# Command to run Streamlit with SSL
CMD streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false --server.sslCertFile=/etc/letsencrypt/live/triathlon.itit.gu.se/cert.pem --server.sslKeyFile=/etc/letsencrypt/live/triathlon.itit.gu.se/privkey.pem
#CMD streamlit run app.py --server.enableCORS False --server.port 8501 --server.sslCertFile=/app/cert.pem --server.sslKeyFile=/app/key.pem

# Run streamlit when the container launches
#CMD streamlit run app.py --server.port 8501
