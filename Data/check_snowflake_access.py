"""
Simple script to verify Snowflake database access.
Run: python check_snowflake_access.py
"""
import sys

import snowflake.connector

import config


def check_access():
    cfg = config.SNOWFLAKE_CONFIG
    missing = [k for k, v in cfg.items() if k != "role" and not v]
    if missing:
        print(f"Missing .env values: {', '.join(f'SNOWFLAKE_{k.upper()}' for k in missing)}")
        sys.exit(1)

    print("Connecting to Snowflake...")
    print(f"  Account:   {cfg['account']}")
    print(f"  User:      {cfg['user']}")
    print(f"  Warehouse: {cfg['warehouse']}")
    print(f"  Database:  {cfg['database']}")
    print(f"  Schema:    {cfg['schema']}")

    conn = None
    try:
        connect_kwargs = {
            "account": cfg["account"],
            "user": cfg["user"],
            "password": cfg["password"],
            "warehouse": cfg["warehouse"],
            "database": cfg["database"],
            "schema": cfg["schema"],
        }
        if cfg.get("role"):
            connect_kwargs["role"] = cfg["role"]

        conn = snowflake.connector.connect(**connect_kwargs)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                CURRENT_USER() AS user,
                CURRENT_ROLE() AS role,
                CURRENT_WAREHOUSE() AS warehouse,
                CURRENT_DATABASE() AS database,
                CURRENT_SCHEMA() AS schema
            """
        )
        row = cursor.fetchone()
        cursor.close()

        print("\nConnection successful.")
        print(f"  Logged in as:  {row[0]}")
        print(f"  Role:          {row[1]}")
        print(f"  Warehouse:     {row[2]}")
        print(f"  Database:      {row[3]}")
        print(f"  Schema:        {row[4]}")
        return 0

    except Exception as e:
        print(f"\nConnection failed: {e}")
        return 1

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    sys.exit(check_access())
