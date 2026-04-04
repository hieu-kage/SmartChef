import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json

POSTGRES_DB = os.getenv("POSTGRES_DB", "smartchef_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

def get_db_connection():
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
    return conn

def get_recipes_by_ids(recipe_ids: list[str]) -> dict:
    """
    Fetch detailed recipe information for a list of IDs.
    Returns a dictionary mapping ID -> Recipe Data (dict).
    """
    if not recipe_ids:
        return {}
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        query = """
            SELECT id, ten_mon, mo_ta, nguyen_lieu_search, nguyen_lieu_chi_tiet, cach_lam, thoi_gian_nau, gia_vi, image_url
            FROM recipes
            WHERE id = ANY(%s)
        """
        cursor.execute(query, (recipe_ids,))
        rows = cursor.fetchall()
        
        result = {row['id']: row for row in rows}
        return result
    except Exception as e:
        print(f"Error querying Postgres: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()
