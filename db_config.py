import os
from sqlalchemy import create_engine


def get_db_engine():
    user = os.getenv('DB_USER', 'root')
    password = os.getenv('DB_PASSWORD', 'root')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '3306')
    dbname = os.getenv('DB_NAME', 'mosaic_db')
    if not all([user, password, dbname]):
        raise ValueError("DB_USER, DB_PASSWORD, and DB_NAME must be set as environment variables.")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url) 