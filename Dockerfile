# Base image with a recent Python version
FROM python:3.11-slim

# set a working directory
WORKDIR /app

# system deps (if any). you may need git, curl, etc. add later if necessary
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# copy and install project requirements
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy rest of the project
COPY . /app

# expose ports if the project runs servers (none by default)

# default environment variables
ENV PYTHONUNBUFFERED=1 \
    # path for evaluating code; can be overridden at runtime
    TASK_TYPE=alfworld \
    ROLLOUT_TYPE=sequential

# add evals folder to path at startup (same as run_main.py does)
ENV PYTHONPATH="/app/evals:${PYTHONPATH}"

# declare entrypoint, allow additional CLI args
ENTRYPOINT ["python", "run_main.py"]
CMD ["--help"]
