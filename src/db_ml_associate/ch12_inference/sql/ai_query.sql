-- --8<-- [start:ai_query]
SELECT customer_id,
       ai_query(
         endpoint   => 'churn-endpoint',              -- an existing Model Serving endpoint
         request    => named_struct(
                         'tenure_months', tenure_months,
                         'monthly_charges', monthly_charges,
                         'data_usage_gb', data_usage_gb,
                         'support_calls', support_calls,
                         'contract_type', contract_type,
                         'payment_method', payment_method),
         returnType => 'DOUBLE'
       ) AS churn_prediction
FROM ml_associate.tutorial.customer_churn;
-- --8<-- [end:ai_query]
