-- --8<-- [start:create_table_sql]
CREATE TABLE ml_associate.tutorial.customer_features (
  customer_id BIGINT NOT NULL,          -- primary key columns must be NOT NULL
  tenure_months BIGINT,
  monthly_charges DOUBLE,
  support_calls_per_year DOUBLE,
  CONSTRAINT customer_features_pk PRIMARY KEY (customer_id)
);
-- --8<-- [end:create_table_sql]
