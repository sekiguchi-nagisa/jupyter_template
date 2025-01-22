FROM python:3.11-bookworm

RUN apt-get update
RUN apt install -y locales fonts-noto-cjk && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=JST-9
ENV PATH="/root/.local/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.0.1 python3 -

COPY ./sandbox/pyproject.toml ./sandbox/poetry.lock ./
RUN poetry install 

# CMD jupyter lab
CMD poetry run jupyter lab