-- --8<-- [start:add_primary_key_sql]
ALTER TABLE ml_associate.tutorial.customer_features ALTER COLUMN customer_id SET NOT NULL;
ALTER TABLE ml_associate.tutorial.customer_features
  ADD CONSTRAINT customer_features_pk PRIMARY KEY (customer_id);
-- --8<-- [end:add_primary_key_sql]
