CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_datamart_dynamic_slice`
PARTITION BY CALENDAR_DATE
CLUSTER BY ITEM_COMMUNICATIONKEY AS
WITH cte_twin_map AS
  (
    SELECT DISTINCT stat.ITEM_COMMUNICATIONKEY, stat.ITEMOPTION_COMMUNICATIONKEY
    FROM `brain-flash-dev.dagster_attributes.twins_lwg_fashion` twins
    JOIN `brain-flash-dev.psf_mart.datamart_static` stat
    ON twins.item_communicationkey = stat.ITEM_COMMUNICATIONKEY
  ),
  cte_first_corso AS
  (
  SELECT
    itemoptions.ITEMOPTION_COMMUNICATIONKEY,
    MIN(CASE WHEN STOCKHOLDINGCOMPANYID = 2 THEN DATE(item.KNOWN_FROM_UTC, 'Europe/Berlin') END) AS FIRST_CORSO_DATE
  FROM
    `brain-flash-dev.psf_mart_inputs.itemoption_inputs` itemoptions
  LEFT JOIN
    `brain-flash-dev.dagster_mercado.exasol_dim_item` AS item
    USING(ITEM_COMMUNICATIONKEY)
  GROUP BY
    ITEMOPTION_COMMUNICATIONKEY
  HAVING MIN(CASE WHEN STOCKHOLDINGCOMPANYID = 2 THEN DATE(item.KNOWN_FROM_UTC, 'Europe/Berlin') END) IS NOT NULL -- itemoption must have been sold to corso
  ),
  cte_filter AS
  (
    SELECT *
    FROM cte_twin_map map
    JOIN cte_first_corso corso
    USING(ITEMOPTION_COMMUNICATIONKEY)
  ),
  cte_returns AS
  (
  SELECT
    coin.ITEMOPTION_COMMUNICATIONKEY,
    KPI_DATE_CET AS CALENDAR_DATE,
    SUM(KPI_46_RETOUREN_STUECK) AS RETOUREN_STUECK
  FROM
    dagster_finance.exasol_kpi_nachfrageumsatz AS coin
  JOIN
    dagster_mercado.exasol_dim_itemoption_latest AS io
  ON
    coin.ITEMOPTION_COMMUNICATIONKEY = io.ITEMOPTION_COMMUNICATIONKEY
  JOIN
    dagster_mercado.v_dim_item AS i
  ON
    io.ITEM_COMMUNICATIONKEY = i.ITEM_COMMUNICATIONKEY
    AND TIMESTAMP(KPI_DATE_CET) BETWEEN i.KNOWN_FROM_UTC
    AND i.KNOWN_UNTIL_UTC
  WHERE
    STOCKHOLDINGCOMPANYID = 0
    AND BUSINESSPROCESSOWNER_ID NOT IN (3, 6)
    AND KPI_46_RETOUREN_STUECK < 0
    --AND KPI_DATE_CET >= DATE_SUB(CURRENT_DATE-1, INTERVAL 12 MONTH)
  GROUP BY
    ITEMOPTION_COMMUNICATIONKEY,
    KPI_DATE_CET
  )

SELECT
  dyn.*, --sollte auch mal eingeschränkt werden auf die wichtigsten
  io_base.ITEM_COMMUNICATIONKEY,
  cte_returns.RETOUREN_STUECK,
  LAST_VALUE(ANSPRACHE_UNBIASED IGNORE NULLS) OVER (ORDER BY dyn.CALENDAR_DATE ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ANSPRACHE_IMPUTED_FFILL -- ist noch nicht live
FROM `brain-flash-dev.psf_mart_inputs.datamart_dynamic` dyn
RIGHT JOIN cte_twin_map io_base
USING(ITEMOPTION_COMMUNICATIONKEY)
JOIN cte_returns
ON dyn.ITEMOPTION_COMMUNICATIONKEY = cte_returns.ITEMOPTION_COMMUNICATIONKEY
AND dyn.CALENDAR_DATE = cte_returns.CALENDAR_DATE
WHERE DATE_DIFF(CURRENT_DATE(), FIRST_ANSPRACHE_DATE, DAY) > 364 --sollte ich das mal fix setzen, dass ich nicht immer neue Ergebnisse bekomme?

-- we assume when one itemoption has been to corso the entire options belonging to the item were sold
-- hätte noch die Spalten im neuen Datamart dynamic einschränken sollen für effizientere queries