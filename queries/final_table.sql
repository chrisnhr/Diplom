CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_final_table` AS
WITH cte_items AS (
  SELECT DISTINCT
    ITEM_COMMUNICATIONKEY AS ITEM_COMMUNICATIONKEY,
    ITEM_COMMUNICATIONKEY AS TWIN_ITEM_COMMUNICATIONKEY,
    NULL AS distance
  FROM `brain-flash-dev.dagster_common.CN_best_twins_mapping`
),
cte_map_all AS (
  SELECT * 
  FROM `brain-flash-dev.dagster_common.CN_best_twins_mapping`
  UNION ALL
  SELECT *
  FROM cte_items
  ORDER BY ITEM_COMMUNICATIONKEY
),
cte_grouped AS (
  SELECT
    cte_map_all.ITEM_COMMUNICATIONKEY,
    cte_map_all.TWIN_ITEM_COMMUNICATIONKEY,
    demand.CALENDAR_DATE,
    SUM(ANSPRACHE) AS ANSPRACHE
  FROM cte_map_all
  JOIN `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice` demand
    ON cte_map_all.TWIN_ITEM_COMMUNICATIONKEY = demand.ITEM_COMMUNICATIONKEY
  GROUP BY cte_map_all.ITEM_COMMUNICATIONKEY, cte_map_all.TWIN_ITEM_COMMUNICATIONKEY, demand.CALENDAR_DATE
),
cte_twin_counts AS (
  SELECT
    ITEM_COMMUNICATIONKEY,
    COUNT(DISTINCT TWIN_ITEM_COMMUNICATIONKEY) AS twin_count
  FROM cte_grouped
  GROUP BY ITEM_COMMUNICATIONKEY
)
SELECT
  cte_grouped.ITEM_COMMUNICATIONKEY,
  cte_grouped.TWIN_ITEM_COMMUNICATIONKEY,
  cte_grouped.CALENDAR_DATE,
  cte_grouped.ANSPRACHE,
  cte_twin_counts.twin_count -1 AS TWIN_COUNT -- weil Testartikel kein Twin ist
FROM cte_grouped
JOIN cte_twin_counts
  ON cte_grouped.ITEM_COMMUNICATIONKEY = cte_twin_counts.ITEM_COMMUNICATIONKEY
ORDER BY cte_twin_counts.twin_count DESC, cte_grouped.ITEM_COMMUNICATIONKEY, cte_grouped.TWIN_ITEM_COMMUNICATIONKEY, cte_grouped.CALENDAR_DATE;
