FROM python:3.12-slim-bookworm

RUN pip install playwright==1.51.0 && \
    playwright install --with-deps

RUN apt update
RUN apt install -y curl wget sudo gcc git libpq-dev autoconf make libtool build-essential

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy the project into the image
ADD . /workspace

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /workspace
RUN uv sync --frozen

CMD [ "tail", "-f", "/dev/null" ]