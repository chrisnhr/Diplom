CREATE TABLE `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice_new`
PARTITION BY CALENDAR_DATE
CLUSTER BY ITEM_COMMUNICATIONKEY AS
WITH
  cte_filter_soldout AS (
    SELECT
      ITEM_COMMUNICATIONKEY,
      CALENDAR_DATE,
      ROW_NUMBER() OVER (PARTITION BY ITEM_COMMUNICATIONKEY ORDER BY CALENDAR_DATE ASC) AS ROWN
    FROM
      `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
    WHERE
      fraction_SOLDOUT > 0
  ),
  cte_first_soldout AS (
    SELECT
      ITEM_COMMUNICATIONKEY,
      dyn.CALENDAR_DATE,
      dyn.FIRST_ANSPRACHE_DATE,
      fraction_SOLDOUT,
      soldout_date.CALENDAR_DATE AS FIRST_SOLDOUT_DATE
    FROM
      `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice` dyn
    JOIN (
      SELECT
        ITEM_COMMUNICATIONKEY,
        CALENDAR_DATE
      FROM
        cte_filter_soldout
      WHERE
        ROWN = 1
    ) soldout_date
    USING
      (ITEM_COMMUNICATIONKEY)
  ),
  cte_soldout_dates AS (
    SELECT
      ITEM_COMMUNICATIONKEY,
      CALENDAR_DATE,
      MIN(cte_first_soldout.FIRST_SOLDOUT_DATE) AS FIRST_SOLDOUT_DATE
    FROM
      cte_first_soldout
    GROUP BY
      ITEM_COMMUNICATIONKEY,
      CALENDAR_DATE
  )

SELECT
  dyn.*,
  cte_first_soldout.FIRST_SOLDOUT_DATE
FROM 
  `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice` dyn
JOIN 
  cte_first_soldout
ON 
  dyn.ITEM_COMMUNICATIONKEY = cte_first_soldout.ITEM_COMMUNICATIONKEY
AND 
  dyn.CALENDAR_DATE = cte_first_soldout.CALENDAR_DATE;

  -- must be added to dynamic slice