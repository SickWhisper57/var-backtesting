CREATE TABLE IF NOT EXISTS var_data (
    id SERIAL PRIMARY KEY,
    index_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    pnl NUMERIC(20, 6),
    var_99 NUMERIC(20, 6),
    CONSTRAINT uq_index_date UNIQUE (index_name, date)
);

CREATE INDEX IF NOT EXISTS idx_index_name_date ON var_data (index_name, date DESC);
