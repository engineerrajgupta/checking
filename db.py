# db.py

import os
import psycopg2.pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# This global variable will hold our connection pool
db_pool = None

def init_db_pool():
    """This function creates the connection pool when the app starts."""
    global db_pool
    if DATABASE_URL and not db_pool:
        try:
            print("Initializing database connection pool...")
            db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5, # A small pool is enough for the free tier
                dsn=DATABASE_URL
            )
            print("Database connection pool initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize database pool: {e}")

def close_db_pool():
    """This function closes all connections when the app shuts down."""
    global db_pool
    if db_pool:
        db_pool.closeall()
        print("Database connection pool closed.")
