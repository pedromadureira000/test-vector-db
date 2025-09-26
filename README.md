# Tutorial
```bash
git clone git@github.com:pedromadureira000/test-vector-db.git
cd test-vector-db
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp contrib/env-sample .env
# Make sure to add your OPENAI_API_KEY to the .env file
# if necessary: `sudo docker compose up -d`
# or: `sudo docker compose up`
source .venv/bin/activate && sudo systemctl start docker
psql postgres://admin_user:asdf@localhost:5432/postgres
create database test_vector_db;
\c test_vector_db
CREATE EXTENSION IF NOT EXISTS vector;
exit;

python runpoc.py
```
