FROM python:3.12-slim
WORKDIR /home/app/src
# Copy and install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

RUN pip install gunicorn
# Copy the source code
COPY . .
# Compile Python source code to bytecode
RUN python -m compileall -b . && \
    find . -type f -name "*.py" -delete
# Expose port
EXPOSE 80
# Define the entrypoint and command
CMD ["gunicorn", "--bind", "0.0.0.0:80", "wsgi:app"]
