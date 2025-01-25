CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_twin_selection` AS
WITH cte_test_first_date AS
  (SELECT
    ITEM_COMMUNICATIONKEY,
    AVG(AVG_ANSPRACHE_AVLBL_OTTO_DE_365) AVG_AVG_ANSPRACHE_AVLBL_OTTO_DE_365, -- in case there is more logic behind the averaging
    MIN(FIRST_ANSPRACHE_DATE) MIN_FIRST_ANSPRACHE_ITEM--what if there are itemoptions added over time?
  FROM `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
  GROUP BY ITEM_COMMUNICATIONKEY
  ORDER BY AVG(AVG_ANSPRACHE_AVLBL_OTTO_DE_365) DESC
  LIMIT(50)),
  cte_twin_dates AS(--Assuming the twins are in the same cleaned_product_group
    SELECT 
      ITEM_COMMUNICATIONKEY,
      MIN(FIRST_ANSPRACHE_DATE) AS TWIN_FIRST_ANSPRACHE_DATE,
      MAX(LAST_ANSPRACHE_DATE) AS TWIN_LAST_ANSPRACHE_DATE
      FROM `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
      GROUP BY ITEM_COMMUNICATIONKEY
  ),
  cte_date_filter AS(
    SELECT
      cte_test_first_date.ITEM_COMMUNICATIONKEY,
      twins.twin_item_communicationkey AS TWIN_ITEM_COMMUNICATIONKEY,
      cte_test_first_date.MIN_FIRST_ANSPRACHE_ITEM AS FIRST_ANSPRACHE_ITEM,
      twin_dates.TWIN_FIRST_ANSPRACHE_DATE,
      DATE_DIFF(cte_test_first_date.MIN_FIRST_ANSPRACHE_ITEM, twin_dates.TWIN_FIRST_ANSPRACHE_DATE, DAY) DIFF_DATE,
      twins.distance
    FROM cte_test_first_date
      JOIN `brain-flash-dev.dagster_attributes.twins_lwg_fashion` twins
        ON cte_test_first_date.ITEM_COMMUNICATIONKEY = twins.item_communicationkey
      JOIN cte_twin_dates twin_dates
        ON twins.twin_item_communicationkey = twin_dates.ITEM_COMMUNICATIONKEY
    WHERE
      DATE_DIFF(cte_test_first_date.MIN_FIRST_ANSPRACHE_ITEM, twin_dates.TWIN_FIRST_ANSPRACHE_DATE, DAY) > 364 -- ensure twin had first demand at least one year before the test article
      AND DATE_DIFF(cte_test_first_date.MIN_FIRST_ANSPRACHE_ITEM, twin_dates.TWIN_LAST_ANSPRACHE_DATE, DAY) < 364 -- ensure twin had demand in the year before the the test article started
  ),
  cte_ranked AS (
    SELECT
      ITEM_COMMUNICATIONKEY,
      TWIN_ITEM_COMMUNICATIONKEY,
      distance,
      ROW_NUMBER() OVER (PARTITION BY ITEM_COMMUNICATIONKEY ORDER BY ABS(distance) ASC) AS rank --Assuming the distance is symmetric -- try order distance!!
      FROM cte_date_filter
  )
SELECT
  ITEM_COMMUNICATIONKEY,
  TWIN_ITEM_COMMUNICATIONKEY,
  distance
FROM cte_ranked
WHERE rank <= 10
ORDER BY ITEM_COMMUNICATIONKEY, distance