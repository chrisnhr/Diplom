CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_twins_dynamic` AS

WITH
  cte_filter AS (
    SELECT static.ITEMOPTION_COMMUNICATIONKEY,
      static.ITEM_COMMUNICATIONKEY
    from `brain-flash-dev.psf_mart.datamart_static`  static
    JOIN `brain-flash-dev.dagster_common.CN_testartikel_10_twins` twins
    ON static.ITEM_COMMUNICATIONKEY = twins.twin_item_communicationkey
  )

SELECT dyn.CALENDAR_DATE,
  SUM(dyn.ANSPRACHE) AS ANSPRACHE_ITEM,
  cf.ITEM_COMMUNICATIONKEY
FROM `brain-flash-dev.psf_mart.v_datamart_dynamic` dyn
RIGHT JOIN 
 cte_filter cf
ON 
 dyn.ITEMOPTION_COMMUNICATIONKEY = cf.ITEMOPTION_COMMUNICATIONKEY
WHERE
  CALENDAR_DATE > "2021-01-01"
GROUP BY cf.ITEM_COMMUNICATIONKEY, dyn.CALENDAR_DATE
ORDER BY 
  cf.ITEM_COMMUNICATIONKEY, dyn.CALENDAR_DATE;

-- Availability etc berücksichtigen?