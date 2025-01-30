WITH
  cte_filter_soldout AS (
  SELECT
    ITEM_COMMUNICATIONKEY,
    CALENDAR_DATE,
    ROW_NUMBER() OVER (PARTITION BY ITEM_COMMUNICATIONKEY ORDER BY CALENDAR_DATE ASC) AS ROWN
  FROM
    `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
  WHERE
    fraction_SOLDOUT > 0 ),
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
      ROWN = 1) soldout_date
  USING
    (ITEM_COMMUNICATIONKEY) )
SELECT
  ITEM_COMMUNICATIONKEY,
  CALENDAR_DATE,
  AVG(cte_first_soldout.fraction_SOLDOUT) AS SOLDOUT_PERC,
  MIN(cte_first_soldout.FIRST_SOLDOUT_DATE) AS FIRST_SOLDOUT_DATE
FROM
  cte_first_soldout
GROUP BY
  ITEM_COMMUNICATIONKEY,
  CALENDAR_DATE
ORDER BY
  ITEM_COMMUNICATIONKEY,
  CALENDAR_DATE