# db.py
import os
import urllib.parse
from psycopg2 import pool
import psycopg2

db_pool = None

def init_db_pool():
    """
    Initialize a global connection pool using DATABASE_URL env var.
    Expects DATABASE_URL in the form: postgres://user:pass@host:port/dbname
    """
    global db_pool
    if db_pool:
        return

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL not set. DB pool will not be initialized.")
        return

    # psycopg2's pool expects parameters, parse the URL
    try:
        result = urllib.parse.urlparse(database_url)
        username = result.username
        password = result.password
        database = result.path[1:]  # drop leading '/'
        hostname = result.hostname
        port = result.port or 5432

        db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # minconn, maxconn
            user=username,
            password=password,
            host=hostname,
            port=port,
            database=database
        )
        print(f"DB pool initialized to {hostname}:{port}/{database}")
    except Exception as e:
        print(f"Failed to init DB pool: {e}")
        db_pool = None

def close_db_pool():
    global db_pool
    if db_pool:
        try:
            db_pool.closeall()
            print("DB pool closed.")
        except Exception as e:
            print(f"Error closing DB pool: {e}")
        finally:
            db_pool = None
