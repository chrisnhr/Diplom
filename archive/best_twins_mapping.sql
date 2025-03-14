CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_best_twins_mapping` AS
WITH cte_top50_items AS
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
      MIN(FIRST_ANSPRACHE_DATE) AS TWIN_FIRST_ANSPRACHE_DATE
      FROM `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
      GROUP BY ITEM_COMMUNICATIONKEY
  ),
  cte_date_filter AS(
    SELECT
      item_dates.ITEM_COMMUNICATIONKEY,
      twins.twin_item_communicationkey AS TWIN_ITEM_COMMUNICATIONKEY,
      item_dates.MIN_FIRST_ANSPRACHE_ITEM AS FIRST_ANSPRACHE_ITEM,
      twin_dates.TWIN_FIRST_ANSPRACHE_DATE,
      DATE_DIFF(item_dates.MIN_FIRST_ANSPRACHE_ITEM, twin_dates.TWIN_FIRST_ANSPRACHE_DATE, DAY) DIFF_DATE,
      twins.distance
    FROM cte_top50_items item_dates
      JOIN `brain-flash-dev.dagster_attributes.twins_lwg_fashion` twins
        ON item_dates.ITEM_COMMUNICATIONKEY = twins.item_communicationkey
      JOIN cte_twin_dates twin_dates
        ON twins.twin_item_communicationkey = twin_dates.ITEM_COMMUNICATIONKEY
    WHERE
      DATE_DIFF(item_dates.MIN_FIRST_ANSPRACHE_ITEM, twin_dates.TWIN_FIRST_ANSPRACHE_DATE, DAY) > 364 --500 rows left
  ),
  cte_ranked AS (
    SELECT
      ITEM_COMMUNICATIONKEY,
      TWIN_ITEM_COMMUNICATIONKEY,
      distance,
      ROW_NUMBER() OVER (PARTITION BY ITEM_COMMUNICATIONKEY ORDER BY ABS(distance) ASC) AS rank --Assuming the distance is symmetric
      FROM cte_date_filter
  )
SELECT
  ITEM_COMMUNICATIONKEY,
  TWIN_ITEM_COMMUNICATIONKEY,
  distance
FROM cte_ranked
WHERE rank <= 10
ORDER BY ITEM_COMMUNICATIONKEY, distance --263 left
