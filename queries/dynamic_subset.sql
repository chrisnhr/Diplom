CREATE OR REPLACE TABLE
  `brain-flash-dev.dagster_common.CN_datamart_dynamic_subset`
PARTITION BY
  CALENDAR_DATE
CLUSTER BY
  ITEM_COMMUNICATIONKEY AS
WITH
  cte_all_keys AS (
  SELECT
    DISTINCT item_communicationkey AS ITEM_COMMUNICATIONKEY
  FROM
    `brain-flash-dev.dagster_attributes.twins_lwg_fashion`
  UNION DISTINCT
  SELECT
    DISTINCT twin_item_communicationkey AS ITEM_COMMUNICATIONKEY
  FROM
    `brain-flash-dev.dagster_attributes.twins_lwg_fashion` ),
  cte_map_static AS (
  SELECT
    keys.ITEM_COMMUNICATIONKEY,
    static.ITEMOPTION_COMMUNICATIONKEY
  FROM
    cte_all_keys keys
  JOIN
    `brain-flash-dev.psf_mart.datamart_static` static
  USING
    (ITEM_COMMUNICATIONKEY)
  ORDER BY
    ITEM_COMMUNICATIONKEY,
    ITEMOPTION_COMMUNICATIONKEY )
SELECT
  *
FROM
  cte_map_static
JOIN
  `brain-flash-dev.psf_mart_inputs.datamart_dynamic`
USING
  (ITEMOPTION_COMMUNICATIONKEY)