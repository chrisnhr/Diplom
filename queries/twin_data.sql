CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_twin_data` AS
WITH cte_first_ansprache AS (
   SELECT
     ITEM_COMMUNICATIONKEY,
     MIN(FIRST_ANSPRACHE_DATE) AS TEST_FIRST_ANSPRACHE
    FROM `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
    GROUP BY ITEM_COMMUNICATIONKEY
),
cte_twin_data AS (
  SELECT
    twin.ITEM_COMMUNICATIONKEY,
    twin.TWIN_ITEM_COMMUNICATIONKEY,
    dyn.CALENDAR_DATE,
    MIN(cte_first_ansprache.TEST_FIRST_ANSPRACHE) AS TEST_FIRST_ANSPRACHE,
    SUM(dyn.ANSPRACHE) AS ANSPRACHE,
    AVG(dyn.fraction_SOLDOUT) AS SOLDOUT_PERC,
  FROM `brain-flash-dev.dagster_common.CN_twin_selection` twin
    JOIN `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice` dyn
    ON twin.TWIN_ITEM_COMMUNICATIONKEY = dyn.ITEM_COMMUNICATIONKEY
    JOIN cte_first_ansprache 
    ON twin.ITEM_COMMUNICATIONKEY = cte_first_ansprache.ITEM_COMMUNICATIONKEY
    GROUP BY ITEM_COMMUNICATIONKEY, TWIN_ITEM_COMMUNICATIONKEY, CALENDAR_DATE
    HAVING CALENDAR_DATE < TEST_FIRST_ANSPRACHE -- cut twin time series as 
    ORDER BY twin.ITEM_COMMUNICATIONKEY, twin.TWIN_ITEM_COMMUNICATIONKEY, dyn.CALENDAR_DATE
),
cte_distinct_test_items AS (
  SELECT DISTINCT
    ITEM_COMMUNICATIONKEY AS ITEM_COMMUNICATIONKEY,
    ITEM_COMMUNICATIONKEY AS TWIN_ITEM_COMMUNICATIONKEY,
  FROM `brain-flash-dev.dagster_common.CN_twin_selection`
),
cte_all_date AS (
  SELECT
    test.ITEM_COMMUNICATIONKEY,
    test.TWIN_ITEM_COMMUNICATIONKEY,
    dyn.CALENDAR_DATE,
    SUM(ANSPRACHE) AS ANSPRACHE,
    AVG(fraction_SOLDOUT) AS SOLDOUT_PERC
  FROM cte_distinct_test_items test
  JOIN `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice` dyn
    ON test.ITEM_COMMUNICATIONKEY = dyn.ITEM_COMMUNICATIONKEY
  GROUP BY test.ITEM_COMMUNICATIONKEY, test.TWIN_ITEM_COMMUNICATIONKEY, dyn.CALENDAR_DATE
  UNION ALL
  SELECT
  ITEM_COMMUNICATIONKEY,
  TWIN_ITEM_COMMUNICATIONKEY,
  CALENDAR_DATE,
  ANSPRACHE,
  SOLDOUT_PERC
  FROM cte_twin_data
),
cte_twin_counts AS (
    SELECT
    ITEM_COMMUNICATIONKEY,
    COUNT(DISTINCT TWIN_ITEM_COMMUNICATIONKEY) AS twin_count
  FROM cte_all_date
  GROUP BY ITEM_COMMUNICATIONKEY
)
SELECT
  cte_all_date.ITEM_COMMUNICATIONKEY AS TEST_ITEM_COMMUNICATIONKEY, --müsste ich iwo früher schon durchziehen für verständlichkeit
  cte_all_date.TWIN_ITEM_COMMUNICATIONKEY,
  cte_all_date.CALENDAR_DATE,
  cte_all_date.ANSPRACHE,
  cte_all_date.SOLDOUT_PERC,
  cte_twin_counts.twin_count -1 AS TWIN_COUNT -- weil Testartikel kein Twin ist
FROM cte_all_date
JOIN cte_twin_counts
  ON cte_all_date.ITEM_COMMUNICATIONKEY = cte_twin_counts.ITEM_COMMUNICATIONKEY
ORDER BY cte_twin_counts.twin_count DESC, cte_all_date.ITEM_COMMUNICATIONKEY, cte_all_date.TWIN_ITEM_COMMUNICATIONKEY, cte_all_date.CALENDAR_DATE;
