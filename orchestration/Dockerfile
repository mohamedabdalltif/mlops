# Use the official Airflow image
FROM apache/airflow:2.9.0

# Copy the requirements file into the image
COPY requirements.txt .

# Install the requirements at build-time
# This ensures they are always available and doesn't slow down startup
RUN pip install --no-cache-dir -r requirements.txt
