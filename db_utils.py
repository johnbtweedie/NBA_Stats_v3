import sqlite3


def upsert_to_sql(df, conn, table_name, pk_cols):
    '''
    write df to table_name, replacing any existing rows that share the same pk_cols
    avoids reading the whole existing table into pandas and rewriting it on every call
    '''
    df = df.reset_index()
    # an upstream join/fan-out can hand us more than one row for the same key in a
    # single call - keep only the last one so we never write duplicates ourselves
    df = df.drop_duplicates(subset=pk_cols, keep='last')

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cur.fetchone() is not None

    if not exists:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    else:
        df.to_sql('_tmp_upsert', conn, if_exists='replace', index=False)
        cols = ', '.join(f'"{c}"' for c in df.columns)
        try:
            cur.execute(f'INSERT OR REPLACE INTO {table_name} ({cols}) SELECT {cols} FROM _tmp_upsert')
            conn.commit()
        finally:
            cur.execute('DROP TABLE _tmp_upsert')
            conn.commit()

    _ensure_unique_index(conn, table_name, pk_cols)


def _ensure_unique_index(conn, table_name, pk_cols):
    '''
    make sure a unique index on pk_cols exists, regardless of how/when the table was
    first created - a table created before this constraint existed (or by other code)
    would otherwise silently let INSERT OR REPLACE degrade into plain INSERT forever
    '''
    cur = conn.cursor()
    index_name = f'idx_{table_name}_pk'
    pk_list = ', '.join(pk_cols)
    try:
        cur.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table_name} ({pk_list})')
        conn.commit()
    except sqlite3.IntegrityError:
        # duplicate rows already in the table (e.g. accumulated before it had a unique
        # index) are blocking creation - keep the most recent copy of each key and retry
        print(f'{table_name} has duplicate ({pk_list}) rows blocking its unique index - deduplicating')
        cur.execute(f'''
            DELETE FROM {table_name}
            WHERE rowid NOT IN (
                SELECT MAX(rowid) FROM {table_name} GROUP BY {pk_list}
            )
        ''')
        conn.commit()
        cur.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table_name} ({pk_list})')
        conn.commit()
