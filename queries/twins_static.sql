--Plan:
-- Query: Testartikel, welche mindestens ein Jahr Historie haben. (Passt er zur Story? (Fashion))
--> twins_selection
-- Query: Haben wir ausreichen (gute) Twins, die 2 Jahre alt sind?
--> twins_static (top 10 twins (distance))
-- Query: datamart dynamic filtern nach den testarte
--> twins_dynamic 
-- twins ggf filern auf gleiche marke
CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_twins_static` AS

WITH
  cte_twins AS (
  SELECT
    *
  FROM
    `brain-flash-dev.dagster_attributes.twins_lwg_fashion`
  WHERE
    item_communicationkey IN (1611665929) --hier um weitere Testartikel ergänzen
  ),
  cte_first_ansprache AS (
  SELECT
    io.ITEM_COMMUNICATIONKEY,
    MIN(CALENDAR_DATE) AS FIRST_ANSPRACHE
  FROM
    `brain-flash-prd.dagster_finance.v_itemoption_ansprache` coin
  JOIN
    `brain-flash-prd.dagster_mercado.exasol_dim_itemoption_latest` io
  ON
    io.ITEMOPTION_COMMUNICATIONKEY = coin.ITEMOPTION_COMMUNICATIONKEY
  WHERE
    ANSPRACHE > 0
  GROUP BY
    io.ITEM_COMMUNICATIONKEY
  )
SELECT
  first_ansprache_test_artikel.FIRST_ANSPRACHE,
  t.*,
  first_ansprache_twin.FIRST_ANSPRACHE AS FIRST_ANSPRACHE_TWIN
FROM
  cte_twins t
JOIN
  cte_first_ansprache first_ansprache_twin
ON
  first_ansprache_twin.ITEM_COMMUNICATIONKEY = t.twin_item_communicationkey
JOIN
  cte_first_ansprache first_ansprache_test_artikel
ON
  first_ansprache_test_artikel.ITEM_COMMUNICATIONKEY = t.item_communicationkey
WHERE
  DATE_DIFF(first_ansprache_test_artikel.FIRST_ANSPRACHE, first_ansprache_twin.FIRST_ANSPRACHE, DAY) > 364
ORDER BY 
  t.distance DESC
LIMIT 10;