CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_final_table` AS
WITH cte_items AS(
  SELECT DISTINCT
    ITEM_COMMUNICATIONKEY AS ITEM_COMMUNICATIONKEY,
    ITEM_COMMUNICATIONKEY AS TWIN_ITEM_COMMUNICATIONKEY,
    NULL AS distance
  FROM `brain-flash-dev.dagster_common.CN_best_twins_mapping`
  ),
  cte_map_all AS(
    SELECT * 
    FROM `brain-flash-dev.dagster_common.CN_best_twins_mapping`
    UNION ALL
    SELECT *
    FROM cte_items
    ORDER By ITEM_COMMUNICATIONKEY
  )
SELECT
  cte_map_all.ITEM_COMMUNICATIONKEY,
  cte_map_all.TWIN_ITEM_COMMUNICATIONKEY,
  demand.CALENDAR_DATE,
  SUM(ANSPRACHE) AS ANSPRACHE
FROM cte_map_all
JOIN `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice` demand
  ON cte_map_all.ITEM_COMMUNICATIONKEY = demand.ITEM_COMMUNICATIONKEY
GROUP BY cte_map_all.ITEM_COMMUNICATIONKEY, cte_map_all.TWIN_ITEM_COMMUNICATIONKEY, demand.CALENDAR_DATE
ORDER BY cte_map_all.ITEM_COMMUNICATIONKEY, cte_map_all.TWIN_ITEM_COMMUNICATIONKEY, demand.CALENDAR_DATE