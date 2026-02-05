import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json

DB_NAME = os.getenv("DB_NAME", "smartchef_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
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
            SELECT id, ten_mon, mo_ta, nguyen_lieu_search, nguyen_lieu_chi_tiet, cach_lam, thoi_gian_nau, gia_vi
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
