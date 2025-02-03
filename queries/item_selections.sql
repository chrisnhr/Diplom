CREATE OR REPLACE TABLE
  `brain-flash-dev.dagster_common.CN_item_selections` AS
WITH
  cte_twin_clean AS (
  SELECT
    *
  FROM
    `brain-flash-dev.dagster_attributes.twins_lwg_fashion`
  WHERE
    item_communicationkey != twin_item_communicationkey ),
  cte_test_selection AS (
  SELECT
    ITEM_COMMUNICATIONKEY,
    AVG(subset.AVG_ANSPRACHE_UNBIASED_AVLBL_OTTO_DE) AS AVG_ANSPRACHE_UNBIASED_AVLBL_OTTO_DE
  FROM
    `brain-flash-dev.dagster_common.CN_datamart_dynamic_subset_ext` subset
  WHERE
    AVG_ANSPRACHE_UNBIASED_AVLBL_OTTO_DE > 0.1
    AND DATE_DIFF(FIRST_SOLDOUT_DATE, FIRST_ANSPRACHE_DATE, DAY) >= 90
  GROUP BY
    ITEM_COMMUNICATIONKEY
  ORDER BY
    AVG_ANSPRACHE_UNBIASED_AVLBL_OTTO_DE DESC),
  cte_dates AS(
  SELECT
    ITEM_COMMUNICATIONKEY,
    MIN(FIRST_ANSPRACHE_DATE) FIRST_ANSPRACHE_DATE,
    MAX(LAST_ANSPRACHE_DATE) LAST_ANSPRACHE_DATE,
  FROM
    `brain-flash-dev.dagster_common.CN_datamart_dynamic_subset_ext`
  GROUP BY
    ITEM_COMMUNICATIONKEY ),
  cte_twin_selection AS (
  SELECT
    test.ITEM_COMMUNICATIONKEY AS TEST_ITEM_COMMUNICATIONKEY,
    twin.twin_item_communicationkey AS TWIN_ITEM_COMMUNICATIONKEY,
    twin.distance AS DISTANCE,
    test_dates.FIRST_ANSPRACHE_DATE AS TEST_FIRST_ANSPRACHE,
    test_dates.LAST_ANSPRACHE_DATE AS TEST_LAST_ANSPRACHE,
    twin_dates.FIRST_ANSPRACHE_DATE AS TWIN_LAST_ANSPRACHE,
    twin_dates.LAST_ANSPRACHE_DATE AS TWIN_LAST_ANSPRACHE
  FROM
    cte_test_selection test
  JOIN
    cte_twin_clean twin
  USING
    (ITEM_COMMUNICATIONKEY)
  LEFT JOIN
    cte_dates test_dates
  USING
    (ITEM_COMMUNICATIONKEY)
  LEFT JOIN
    cte_dates twin_dates
  ON
    twin.twin_item_communicationkey = twin_dates.ITEM_COMMUNICATIONKEY
  WHERE
    DATE_DIFF(test_dates.FIRST_ANSPRACHE_DATE, twin_dates.FIRST_ANSPRACHE_DATE, DAY) > 364
    AND DATE_DIFF (test_dates.FIRST_ANSPRACHE_DATE, twin_dates.LAST_ANSPRACHE_DATE, DAY) < 364 ),
  cte_twin_ranks AS (
  SELECT
    TEST_ITEM_COMMUNICATIONKEY,
    TWIN_ITEM_COMMUNICATIONKEY,
    DISTANCE,
    cte_twin_selection.TEST_FIRST_ANSPRACHE,
    ROW_NUMBER() OVER (PARTITION BY TEST_ITEM_COMMUNICATIONKEY ORDER BY ABS(DISTANCE) ASC) AS RANK
  FROM
    cte_twin_selection )
SELECT
  TEST_ITEM_COMMUNICATIONKEY,
  TWIN_ITEM_COMMUNICATIONKEY,
  TEST_FIRST_ANSPRACHE,
  DISTANCE
FROM
  cte_twin_ranks
WHERE
  RANK <= 10
ORDER BY
  TEST_ITEM_COMMUNICATIONKEY,
  DISTANCE