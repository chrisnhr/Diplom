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
  )

SELECT *
FROM `brain-flash-dev.psf_mart_inputs.datamart_dynamic` dyn
RIGHT JOIN cte_twin_map io_base
USING(ITEMOPTION_COMMUNICATIONKEY)
WHERE DATE_DIFF(CURRENT_DATE(), FIRST_ANSPRACHE_DATE, DAY) > 364 --sollte ich das mal fix setzen, dass ich nicht immer neue Ergebnisse bekomme?

-- we assume when one itemoption has been to corso the entire options belonging to the item were sold
-- hätte noch die Spalten im neuen Datamart dynamic einschränken sollen für effizientere queries