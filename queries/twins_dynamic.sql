CREATE OR REPLACE TABLE `brain-flash-dev.dagster_common.CN_twins_dynamic` AS

WITH
  cte_filter AS (
    SELECT stat.ITEMOPTION_COMMUNICATIONKEY,
      stat.ITEM_COMMUNICATIONKEY,
      twins.item_communicationkey AS TEST_ITEM_COMMUNICATIONKEY
    from `brain-flash-dev.psf_mart.datamart_static`  stat
    JOIN `brain-flash-dev.dagster_common.CN_twins_static` twins
    ON stat.ITEM_COMMUNICATIONKEY = twins.twin_item_communicationkey
  )

SELECT 
  MAX(cf.TEST_ITEM_COMMUNICATIONKEY) AS TEST_ITEM_COMMUNICATIONKEY, --should be the same throughout the group
  dyn.CALENDAR_DATE,
  SUM(dyn.ANSPRACHE) AS ANSPRACHE_TWIN,
  cf.ITEM_COMMUNICATIONKEY AS TWIN_ITEM_COMMUNICATIONKEY
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