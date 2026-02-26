# Gebruik de officiële Python 3.13 image
FROM python:3.13-slim

# Zet de werkmap in de container
WORKDIR /code

# Kopieer de requirements
COPY requirements.txt .

# Installeer de pakketten
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer je code naar de container
COPY . .

# Start de server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]