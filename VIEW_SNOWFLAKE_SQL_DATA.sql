create or replace view JAFFLE_LINEAGE.LINEAGE_DATA.VIEW_SNOWFLAKE_SQL_DATA(
	QUERY_TEXT,
	TABLE_NAME
) as 
(WITH object_changes AS (
    -- Check if there are object modifications (DML changes)
    SELECT
        qh.query_id,
        qh.query_text,
        mo.value:objectName::STRING AS modified_object,
        'DML' AS change_type,
        rank() over (partition by modified_object order by query_start_time desc) as rnk
    FROM
        snowflake.account_usage.access_history AS ah
        JOIN snowflake.account_usage.query_history AS qh
        ON ah.query_id = qh.query_id
        , LATERAL FLATTEN(input => ah.objects_modified) AS mo
    WHERE
        mo.value:objectName IS NOT NULL
        and ah.user_name = 'JAFFLE_EXECUTOR'
        and (query_text like 'create%or%replace%view%' or query_text like 'create%or%replace%transient%table%' )
    
    UNION
    
    -- Fallback to check for DDL operations (schema changes)
    SELECT
    qh.query_id,
    qh.query_text,
    ddl.value:objectName AS modified_object,  -- Directly extract the value field
    'DDL' AS change_type,
    rank() over (partition by modified_object order by query_start_time desc) as rnk
FROM
    snowflake.account_usage.access_history AS ah
    JOIN snowflake.account_usage.query_history AS qh
    ON ah.query_id = qh.query_id
    , LATERAL FLATTEN(input => ah.object_modified_by_ddl::array) AS ddl
WHERE
        ddl.value:objectName IS NOT NULL
        and ah.user_name = 'JAFFLE_EXECUTOR'
        and (query_text like 'create%or%replace%view%' or query_text like 'create%or%replace%transient%table%' )

)
SELECT
    distinct
    query_text,
    UPPER(modified_object) as table_name
FROM
    object_changes
    where rnk = 1
);